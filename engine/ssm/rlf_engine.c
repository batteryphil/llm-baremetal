/* rlf_engine.c — Bare-metal RLF reasoning engine
 * Freestanding C, no libc. Uses mamba_block.h primitives.
 */
#include "rlf_engine.h"
#include "mamba_block.h"
#include <stddef.h>
#include <stdint.h>

/* ── F16 → F32 conversion ─────────────────────────────────────────────── */
static rlf_f32 f16_to_f32(rlf_f16 h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t f;
    if (e == 0)       { f = (s << 31) | ((m) << 13); }
    else if (e == 31) { f = (s << 31) | 0x7F800000 | (m << 13); }
    else              { f = (s << 31) | ((e + 112) << 23) | (m << 13); }
    union { uint32_t i; rlf_f32 v; } u; u.i = f; return u.v;
}

/* ── Math primitives ──────────────────────────────────────────────────── */
static rlf_f32 rlf_tanhf(rlf_f32 x) {
    if (x >  5.0f) return  1.0f;
    if (x < -5.0f) return -1.0f;
    rlf_f32 x2 = x*x;
    rlf_f32 n = x*(135135.0f + x2*(17325.0f + x2*(378.0f + x2)));
    rlf_f32 d = 135135.0f + x2*(62370.0f + x2*(3150.0f + x2*28.0f));
    return n/d;
}
static rlf_f32 rlf_gelu(rlf_f32 x) {
    return 0.5f * x * (1.0f + rlf_tanhf(0.7978845608f*(x + 0.044715f*x*x*x)));
}
static rlf_f32 rlf_sqrtf(rlf_f32 x) {
    if (x <= 0.0f) return 0.0f;
    union { rlf_f32 f; uint32_t i; } u; u.f = x;
    u.i = 0x5F375A86 - (u.i >> 1);
    rlf_f32 y = u.f;
    y *= 1.5f - 0.5f*x*y*y;
    y *= 1.5f - 0.5f*x*y*y;
    return x * y;
}

/* ── String helpers (no libc) ─────────────────────────────────────────── */
static int rlf_strlen(const char *s) {
    int n = 0; while (s[n]) n++; return n;
}
static void rlf_strcpy(char *d, const char *s) {
    while ((*d++ = *s++));
}
static int rlf_strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (unsigned char)*a - (unsigned char)*b;
}
static int rlf_isdigit(char c) { return c >= '0' && c <= '9'; }
static int rlf_isalpha(char c) {
    return (c>='a'&&c<='z') || (c>='A'&&c<='Z');
}
static int rlf_isalnum(char c) { return rlf_isalpha(c)||rlf_isdigit(c)||c=='_'; }

static rlf_f32 rlf_atof(const char *s) {
    rlf_f32 r=0.0f, frac=0.0f, fdiv=1.0f; int neg=0, in_frac=0;
    if (*s=='-'){neg=1;s++;}
    while (*s) {
        if (*s=='.'){in_frac=1;s++;continue;}
        if (!rlf_isdigit(*s)) break;
        if (in_frac){fdiv*=10.0f;frac+=(*s-'0')/fdiv;}
        else r=r*10.0f+(*s-'0');
        s++;
    }
    return neg ? -(r+frac) : (r+frac);
}
void rlf_ftoa(rlf_f32 v, char *buf) {
    if (v < 0) { *buf++='-'; v=-v; }
    int iv = (int)v; rlf_f32 fv = v - (rlf_f32)iv;
    /* integer part */
    char tmp[16]; int ti=0;
    if (iv==0) tmp[ti++]='0';
    else { int t=iv; while(t){tmp[ti++]='0'+t%10;t/=10;} }
    for(int i=ti-1;i>=0;i--) *buf++=tmp[i];
    /* fractional part — up to 2 decimal places */
    rlf_f32 eps = 0.005f;
    if (fv > eps) {
        *buf++='.';
        fv *= 10.0f; int d=(int)fv; *buf++='0'+d; fv-=(rlf_f32)d;
        fv *= 10.0f; d=(int)(fv+0.5f); *buf++='0'+d;
    }
    *buf='\0';
}

/* ── Linear layer (F16 weights) ────────────────────────────────────────── */
static void rlf_linear_f16(
    const rlf_f16 *W, const rlf_f16 *b,
    const rlf_f32 *x, rlf_f32 *y,
    int in_dim, int out_dim)
{
    for (int o = 0; o < out_dim; o++) {
        rlf_f32 acc = b ? f16_to_f32(b[o]) : 0.0f;
        const rlf_f16 *row = W + (rlf_ulong)o * in_dim;
        for (int i = 0; i < in_dim; i++)
            acc += f16_to_f32(row[i]) * x[i];
        y[o] = acc;
    }
}

/* ── RMSNorm ────────────────────────────────────────────────────────────── */
static void rlf_rmsnorm(const rlf_f32 *x, const rlf_f32 *w,
                         rlf_f32 *y, int n) {
    rlf_f32 ss = 0.0f;
    for (int i=0;i<n;i++) ss += x[i]*x[i];
    rlf_f32 inv = 1.0f / rlf_sqrtf(ss/(rlf_f32)n + 1e-6f);
    for (int i=0;i<n;i++) y[i] = x[i]*inv*(w ? w[i] : 1.0f);
}

/* ── OOSS loader ────────────────────────────────────────────────────────── */
int rlf_ooss_load(const void *data, rlf_ulong size, RlfWeights *out) {
    const uint8_t *p = (const uint8_t*)data;
    const uint8_t *end = p + size;
    /* magic "OOSS" */
    if (size < 12) return -1;
    if (p[0]!='O'||p[1]!='O'||p[2]!='S'||p[3]!='S') return -2;
    uint32_t version, n_tensors;
    __builtin_memcpy(&version,   p+4, 4);
    __builtin_memcpy(&n_tensors, p+8, 4);
    p += 12;

    /* defaults */
    out->cp_hidden   = RLF_CP_HIDDEN;
    out->d_model     = RLF_D_MODEL;
    out->prefix_m    = RLF_PREFIX_M;
    out->d_bridge    = RLF_D_BRIDGE;
    out->lifeline_gate = 0.7f;

    for (uint32_t ti = 0; ti < n_tensors && p < end; ti++) {
        uint32_t name_len; __builtin_memcpy(&name_len, p, 4); p+=4;
        char name[128]; if(name_len>=128) return -3;
        __builtin_memcpy(name, p, name_len); name[name_len]='\0'; p+=name_len;
        uint32_t n_dims; __builtin_memcpy(&n_dims, p, 4); p+=4;
        uint64_t shape[4]; for(uint32_t d=0;d<n_dims&&d<4;d++){
            __builtin_memcpy(&shape[d],p,8);p+=8;}
        uint32_t dtype; uint64_t dlen;
        __builtin_memcpy(&dtype,p,4);p+=4;
        __builtin_memcpy(&dlen,p,8);p+=8;
        const rlf_f16 *tdata = (const rlf_f16*)p; p+=dlen;

        /* map name → struct field */
        if (!rlf_strcmp(name,"rlf.perceptron.mapper_0_weight")) out->cp_w1=tdata;
        else if (!rlf_strcmp(name,"rlf.perceptron.mapper_0_bias"))  out->cp_b1=tdata;
        else if (!rlf_strcmp(name,"rlf.perceptron.mapper_2_weight")) out->cp_w2=tdata;
        else if (!rlf_strcmp(name,"rlf.perceptron.mapper_2_bias"))   out->cp_b2=tdata;
        else if (!rlf_strcmp(name,"rlf.perceptron.mapper_4_weight")) out->cp_w3=tdata;
        else if (!rlf_strcmp(name,"rlf.perceptron.mapper_4_bias"))   out->cp_b3=tdata;
        else if (!rlf_strcmp(name,"rlf.bridge_down.weight"))         out->bd_w=tdata;
        else if (!rlf_strcmp(name,"rlf.bridge_down.bias"))           out->bd_b=tdata;
        else if (!rlf_strcmp(name,"rlf.bridge_up.weight"))           out->bu_w=tdata;
        else if (!rlf_strcmp(name,"rlf.bridge_up.bias"))             out->bu_b=tdata;
        else if (!rlf_strcmp(name,"rlf.loop_norm.weight")) {
            out->loop_norm_w = (const rlf_f32*)tdata; /* stored F32 */
        }
        else if (!rlf_strcmp(name,"rlf.lifeline.gate")) {
            out->lifeline_gate = f16_to_f32(*tdata);
        }
    }
    return 0;
}

/* ── SymbolicPreprocessor ───────────────────────────────────────────────── */

/* Look up a name in env; return float value or 0 */
static int env_get(const RlfEnv *env, const char *name, rlf_f32 *out) {
    for (int i=0;i<env->n;i++) {
        if (!rlf_strcmp(env->bindings[i].name, name) &&
            env->bindings[i].resolved) {
            *out = env->bindings[i].numeric; return 1;
        }
    }
    return 0;
}

/* Evaluate "VAL" or "A*B" where A,B are numbers or env vars */
static int eval_expr(const char *expr, const RlfEnv *env, rlf_f32 *result) {
    /* find '*' */
    int star = -1;
    for (int i=0; expr[i]; i++) if (expr[i]=='*'){star=i;break;}
    if (star < 0) {
        /* simple: number or var */
        if (rlf_isdigit(expr[0]) || expr[0]=='-' || expr[0]=='.') {
            *result = rlf_atof(expr); return 1;
        }
        return env_get(env, expr, result);
    }
    /* A*B */
    char lhs[32]={0}, rhs[32]={0};
    for(int i=0;i<star&&i<31;i++) lhs[i]=expr[i];
    int ri=0; for(int i=star+1;expr[i]&&ri<31;i++) rhs[ri++]=expr[i];
    rlf_f32 a=0.0f, b=0.0f;
    int al, bl;
    if (rlf_isdigit(lhs[0]) || lhs[0]=='-') { a=rlf_atof(lhs); al=1; }
    else al = env_get(env, lhs, &a);
    if (rlf_isdigit(rhs[0]) || rhs[0]=='.' || rhs[0]=='-') { b=rlf_atof(rhs); bl=1; }
    else bl = env_get(env, rhs, &b);
    if (!al||!bl) return 0;
    *result = a*b; return 1;
}

void rlf_preprocess(const char *prompt, RlfCtx *ctx) {
    RlfEnv *env = &ctx->env;
    env->n = 0;
    char *out = ctx->prompt_pp;
    int olen = 0;
    const int OMAX = RLF_MAX_PROMPT - 1;

    /* Pass 1: scan for "WORD=EXPR." assignments */
    const char *p = prompt;
    while (*p && olen < OMAX) {
        /* try to match identifier */
        if (rlf_isalpha(*p) || *p=='_') {
            const char *id_start = p;
            while (rlf_isalnum(*p)) p++;
            int id_len = (int)(p - id_start);
            if (*p == '=' && id_len > 0 && id_len < 31) {
                /* got VAR= */
                char name[32]={0};
                for(int i=0;i<id_len;i++) name[i]=id_start[i];
                p++; /* skip '=' */
                /* read value until '.' or ' ' or end */
                char val[64]={0}; int vi=0;
                while (*p && *p!='.' && *p!=' ' && *p!='\n' && vi<63)
                    val[vi++]=*p++;
                if (env->n < RLF_MAX_BINDINGS) {
                    RlfBinding *b = &env->bindings[env->n++];
                    rlf_strcpy(b->name, name);
                    rlf_strcpy(b->value, val);
                    b->resolved = 0;
                }
                /* copy "NAME=VAL" to out */
                for(int i=0;i<id_len&&olen<OMAX;i++) out[olen++]=name[i];
                out[olen++]='=';
                for(int i=0;val[i]&&olen<OMAX;i++) out[olen++]=val[i];
                continue;
            }
            /* not an assignment — copy as-is */
            for(const char *q=id_start;q<p&&olen<OMAX;q++) out[olen++]=*q;
            continue;
        }
        out[olen++] = *p++;
    }
    out[olen]='\0';

    /* Pass 2a: resolve literals */
    for(int i=0;i<env->n;i++) {
        RlfBinding *b=&env->bindings[i];
        const char *v=b->value;
        int is_num=0;
        if(rlf_isdigit(v[0])||v[0]=='-'||(v[0]=='.'&&rlf_isdigit(v[1])))
            is_num=1;
        if(is_num && !__builtin_strchr(v,'*')) {
            b->numeric=rlf_atof(v); b->resolved=1;
        }
    }

    /* Pass 2b: resolve arithmetic using complete env */
    for(int i=0;i<env->n;i++) {
        RlfBinding *b=&env->bindings[i];
        if(b->resolved) continue;
        rlf_f32 res;
        if(eval_expr(b->value, env, &res)){
            b->numeric=res; b->resolved=1;
            /* rewrite in out string: find "NAME=OLD" replace value with number */
            char numstr[32]; rlf_ftoa(res,numstr);
            /* simple scan-replace in prompt_pp */
            char *found = out;
            int nlen = rlf_strlen(b->name);
            while(*found) {
                int match=1;
                for(int k=0;k<nlen;k++) if(found[k]!=b->name[k]){match=0;break;}
                if(match && found[nlen]=='=') {
                    char *eq = found+nlen+1;
                    /* overwrite value digits */
                    int old_len=0;
                    while(eq[old_len]&&eq[old_len]!='.'&&eq[old_len]!=' ') old_len++;
                    int new_len=rlf_strlen(numstr);
                    if(new_len<=old_len) {
                        for(int k=0;k<new_len;k++) eq[k]=numstr[k];
                        /* pad with spaces if shorter */
                        for(int k=new_len;k<old_len;k++) eq[k]=' ';
                    }
                    break;
                }
                found++;
            }
        }
    }
}

#ifndef RLF_PREPROCESS_ONLY
/* ── ConceptPerceptron ──────────────────────────────────────────────────── */
void rlf_concept_perceptron(const RlfWeights *w, RlfCtx *ctx) {
    int D = RLF_D_MODEL, H = RLF_CP_HIDDEN, M = RLF_PREFIX_M;
    /* Layer 1: D → H with GELU */
    rlf_linear_f16(w->cp_w1, w->cp_b1, ctx->x_mean,
                   ctx->cp_hidden_buf, D, H);
    for(int i=0;i<H;i++) ctx->cp_hidden_buf[i]=rlf_gelu(ctx->cp_hidden_buf[i]);
    /* Layer 2: H → H with GELU */
    static rlf_f32 tmp[RLF_CP_HIDDEN];
    rlf_linear_f16(w->cp_w2, w->cp_b2, ctx->cp_hidden_buf, tmp, H, H);
    for(int i=0;i<H;i++) tmp[i]=rlf_gelu(tmp[i]);
    /* Layer 3: H → M*D (prefix tokens) */
    rlf_linear_f16(w->cp_w3, w->cp_b3, tmp, ctx->prefix, H, M*D);
    /* L2-normalise each prefix token */
    for(int m=0;m<M;m++){
        rlf_f32 *tok=ctx->prefix+m*D;
        rlf_f32 ss=0.0f;
        for(int i=0;i<D;i++) ss+=tok[i]*tok[i];
        rlf_f32 inv=1.0f/rlf_sqrtf(ss/(rlf_f32)D+1e-8f);
        for(int i=0;i<D;i++) tok[i]*=inv;
    }
}

/* ── Latent bridge ──────────────────────────────────────────────────────── */
void rlf_bridge_encode(const RlfWeights *w, const rlf_f32 *x, RlfCtx *ctx) {
    rlf_linear_f16(w->bd_w, w->bd_b, x, ctx->bridge_latent,
                   RLF_D_MODEL, RLF_D_BRIDGE);
}

void rlf_bridge_decode(const RlfWeights *w, RlfCtx *ctx, rlf_f32 *x_out) {
    rlf_linear_f16(w->bu_w, w->bu_b, ctx->bridge_latent, x_out,
                   RLF_D_BRIDGE, RLF_D_MODEL);
}
#endif /* !RLF_PREPROCESS_ONLY */

/* ── Status printer ─────────────────────────────────────────────────────── */
void rlf_status_print(const RlfWeights *w) {
    /* Uses the REPL print mechanism — declared extern by the REPL layer */
    extern void repl_print(const char *);
    if (!w) { repl_print("[RLF] not loaded\r\n"); return; }
    repl_print("[RLF] loaded\r\n");
    repl_print("  halt_token : 7803 (SS)\r\n");
    repl_print("  max_loops  : 6\r\n");
    repl_print("  prefix_m   : 8\r\n");
    repl_print("  d_bridge   : 128\r\n");
    repl_print("  cp_w1      : "); repl_print(w->cp_w1 ? "OK" : "MISSING"); repl_print("\r\n");
    repl_print("  bd_w       : "); repl_print(w->bd_w  ? "OK" : "MISSING"); repl_print("\r\n");
}

#ifndef RLF_PREPROCESS_ONLY
/* ── Main inference entry ───────────────────────────────────────────────── */
/*
 * rlf_infer wires the preprocessor → concept perceptron → backbone loop.
 * The backbone call (mamba_forward) is provided by the existing GGUF/SSM
 * engine through the interface defined in engine/ssm/interface.h.
 * We declare the extern here; the linker resolves it at build time.
 *
 * Signature expected from interface.h:
 *   int mamba_forward(void *model, const uint32_t *tokens, int n_tokens,
 *                     float *out_hidden, int d_model);
 *   int mamba_lm_head(void *model, const float *hidden,
 *                     uint32_t *top1_token, float *top1_logit);
 *   int mamba_tokenize(void *model, const char *text,
 *                      uint32_t *ids, int max_ids);
 *   void mamba_detokenize(void *model, uint32_t id, char *buf, int len);
 */
extern int mamba_forward(void*, const uint32_t*, int, rlf_f32*, int);
extern int mamba_lm_head(void*, const rlf_f32*, uint32_t*, rlf_f32*);
extern int mamba_tokenize(void*, const char*, uint32_t*, int);
extern void mamba_detokenize(void*, uint32_t, char*, int);

int rlf_infer(
    const char      *prompt,
    void            *backbone,
    const RlfWeights *w,
    RlfCtx          *ctx,
    char            *answer,
    int              answer_len,
    int             *n_loops_out)
{
    int D = RLF_D_MODEL, M = RLF_PREFIX_M;

    /* 1. Symbolic preprocessing */
    rlf_preprocess(prompt, ctx);

    /* 2. Tokenise preprocessed prompt */
    static uint32_t prompt_ids[RLF_MAX_PROMPT];
    int n_prompt = mamba_tokenize(backbone, ctx->prompt_pp,
                                  prompt_ids, RLF_MAX_PROMPT);
    if (n_prompt <= 0) return -1;

    /* 3. Initial backbone pass → x_mean for ConceptPerceptron */
    static rlf_f32 hidden_buf[RLF_MAX_PROMPT * RLF_D_MODEL];
    mamba_forward(backbone, prompt_ids, n_prompt, hidden_buf, D);
    /* mean-pool over sequence */
    for(int d=0;d<D;d++){
        rlf_f32 s=0.0f;
        for(int t=0;t<n_prompt;t++) s+=hidden_buf[(rlf_ulong)t*D+d];
        ctx->x_mean[d]=s/(rlf_f32)n_prompt;
    }

    /* 4. ConceptPerceptron → prefix tokens */
    if (w->cp_w1) rlf_concept_perceptron(w, ctx);

    /* 5. Build prefix+prompt token sequence */
    static uint32_t full_ids[RLF_PREFIX_M + RLF_MAX_PROMPT];
    /* prefix tokens are represented as special token 1 (padding) repeated;
     * the actual prefix embeddings are injected by mamba_forward_with_prefix
     * if available, otherwise we fall back to prompt-only inference */
    for(int m=0;m<M;m++) full_ids[m] = 1; /* placeholder */
    for(int t=0;t<n_prompt;t++) full_ids[M+t] = prompt_ids[t];
    int n_full = M + n_prompt;

    ctx->n_answer = 0;
    answer[0] = '\0';
    int ret = 1; /* default: max loops hit */

    /* 6. RLF loop */
    static rlf_f32 loop_hidden[RLF_D_MODEL];
    static rlf_f32 bridge_out[RLF_D_MODEL];

    for (int loop = 0; loop < RLF_MAX_LOOPS; loop++) {
        /* Run backbone on full sequence */
        mamba_forward(backbone, full_ids, n_full, hidden_buf, D);

        /* Take last token hidden state */
        rlf_f32 *last_h = hidden_buf + (rlf_ulong)(n_full-1)*D;

        /* Apply loop RMSNorm */
        if (w->loop_norm_w)
            rlf_rmsnorm(last_h, w->loop_norm_w, loop_hidden, D);
        else
            for(int d=0;d<D;d++) loop_hidden[d]=last_h[d];

        /* lm_head → argmax */
        uint32_t tok; rlf_f32 logit;
        mamba_lm_head(backbone, loop_hidden, &tok, &logit);

        if (tok == RLF_HALT_TOKEN) {
            ret = 0; /* clean halt */
            break;
        }

        /* Append answer token */
        if (ctx->n_answer < RLF_MAX_ANSWER)
            ctx->answer_tokens[ctx->n_answer++] = tok;

        /* Bridge: encode last hidden → latent → decode → update prefix */
        rlf_bridge_encode(w, loop_hidden, ctx);
        rlf_bridge_decode(w, ctx, bridge_out);

        /* Lifeline: inject bridge output into prefix position 0 */
        rlf_f32 gate = w->lifeline_gate;
        for(int d=0;d<D;d++)
            ctx->prefix[d] = ctx->prefix[d] + gate * bridge_out[d];

        /* Append generated token to sequence for next loop */
        if (n_full < RLF_PREFIX_M + RLF_MAX_PROMPT)
            full_ids[n_full++] = tok;
    }

    if (n_loops_out) *n_loops_out = ctx->n_answer;

    /* 7. Decode answer tokens to string */
    int pos = 0;
    for (int i = 0; i < ctx->n_answer && pos < answer_len-4; i++) {
        char tmp[16];
        mamba_detokenize(backbone, ctx->answer_tokens[i], tmp, 16);
        for(int k=0; tmp[k] && pos < answer_len-1; k++)
            answer[pos++] = tmp[k];
    }
    answer[pos] = '\0';

    return ret;
}
#endif /* !RLF_PREPROCESS_ONLY */
