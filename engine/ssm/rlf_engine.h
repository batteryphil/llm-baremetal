/* rlf_engine.h — Bare-metal RLF reasoning engine public API
 * ==========================================================
 * Freestanding C, no libc. Caller-managed buffers throughout.
 * Compatible with llm-baremetal UEFI runtime (EFI_SYSTEM_TABLE available).
 *
 * Load sequence:
 *   1. Load backbone GGUF via existing gguf_loader
 *   2. Call rlf_ooss_load() with the .ooss sidecar
 *   3. Call rlf_infer() for each prompt
 */

#ifndef RLF_ENGINE_H
#define RLF_ENGINE_H

#include <stdint.h>

/* ── Type aliases ───────────────────────────────────────────────────────── */
typedef float          rlf_f32;
typedef uint16_t       rlf_f16;   /* raw IEEE 754 half — no arithmetic */
typedef unsigned int   rlf_uint;
typedef unsigned long  rlf_ulong;

/* ── Limits ─────────────────────────────────────────────────────────────── */
#define RLF_MAX_LOOPS       6
#define RLF_PREFIX_M        8
#define RLF_D_MODEL         2048
#define RLF_D_BRIDGE        128
#define RLF_HALT_TOKEN      7803   /* § in GPT-NeoX vocab */
#define RLF_MAX_BINDINGS    32     /* SymbolicPreprocessor var table */
#define RLF_MAX_PROMPT      512    /* max prompt chars */
#define RLF_MAX_ANSWER      64     /* max answer chars */
#define RLF_CP_HIDDEN       16384  /* ConceptPerceptron hidden dim */

/* ── OOSS sidecar weight blob ────────────────────────────────────────────── */
typedef struct {
    /* ConceptPerceptron MLP weights (F16, loaded directly) */
    const rlf_f16 *cp_w1;   /* [D_MODEL, CP_HIDDEN]     input projection  */
    const rlf_f16 *cp_b1;   /* [CP_HIDDEN]               bias 1            */
    const rlf_f16 *cp_w2;   /* [CP_HIDDEN, CP_HIDDEN]    hidden layer      */
    const rlf_f16 *cp_b2;   /* [CP_HIDDEN]               bias 2            */
    const rlf_f16 *cp_w3;   /* [CP_HIDDEN, D_MODEL*PREFIX_M] output proj   */
    const rlf_f16 *cp_b3;   /* [D_MODEL*PREFIX_M]        bias 3            */

    /* Latent bridge */
    const rlf_f16 *bd_w;    /* [D_MODEL, D_BRIDGE]  down projection        */
    const rlf_f16 *bd_b;    /* [D_BRIDGE]            down bias              */
    const rlf_f16 *bu_w;    /* [D_BRIDGE, D_MODEL]  up projection           */
    const rlf_f16 *bu_b;    /* [D_MODEL]             up bias                */

    /* Loop RMSNorm weight */
    const rlf_f32 *loop_norm_w;  /* [D_MODEL] */

    /* Lifeline gate scalar */
    rlf_f32 lifeline_gate;

    /* Convenience dims (filled by rlf_ooss_load) */
    rlf_uint cp_hidden;      /* should equal RLF_CP_HIDDEN */
    rlf_uint d_model;        /* should equal RLF_D_MODEL   */
    rlf_uint prefix_m;       /* should equal RLF_PREFIX_M  */
    rlf_uint d_bridge;       /* should equal RLF_D_BRIDGE  */
} RlfWeights;

/* ── SymbolicPreprocessor binding ────────────────────────────────────────── */
typedef struct {
    char name[32];
    char value[32];          /* resolved numeric string or symbolic name */
    int  resolved;           /* 1 = numeric, 0 = still symbolic          */
    rlf_f32 numeric;         /* valid when resolved == 1                 */
} RlfBinding;

typedef struct {
    RlfBinding bindings[RLF_MAX_BINDINGS];
    int        n;
} RlfEnv;

/* ── Inference context (caller-allocated scratch) ────────────────────────── */
typedef struct {
    /* Scratchpad for ConceptPerceptron forward pass */
    rlf_f32 cp_hidden_buf[RLF_CP_HIDDEN];
    /* Prefix token embeddings: [PREFIX_M × D_MODEL] */
    rlf_f32 prefix[RLF_PREFIX_M * RLF_D_MODEL];
    /* Bridge latent: [D_BRIDGE] */
    rlf_f32 bridge_latent[RLF_D_BRIDGE];
    /* Prompt mean hidden state: [D_MODEL] */
    rlf_f32 x_mean[RLF_D_MODEL];
    /* Single token hidden state after norm: [D_MODEL] */
    rlf_f32 x_normed[RLF_D_MODEL];
    /* Answer token buffer */
    uint32_t answer_tokens[RLF_MAX_ANSWER];
    int      n_answer;
    /* Preprocessed prompt string */
    char     prompt_pp[RLF_MAX_PROMPT];
    /* Resolved env */
    RlfEnv   env;
} RlfCtx;

/* ── API ─────────────────────────────────────────────────────────────────── */

/**
 * rlf_ooss_load - Parse a .ooss sidecar blob into RlfWeights.
 *
 * @data:   Raw bytes of the .ooss file (mapped into memory).
 * @size:   Byte length of data.
 * @out:    Caller-allocated RlfWeights struct to fill.
 *
 * Returns 0 on success, negative on error.
 */
int rlf_ooss_load(const void *data, rlf_ulong size, RlfWeights *out);

/**
 * rlf_preprocess - Run SymbolicPreprocessor on a prompt.
 *
 * Resolves "VAR=NUM", "VAR=VAR2", "VAR=VAR2*NUM", and word-math
 * ("twice", "double", "half") into a cleaned prompt string plus env.
 *
 * @prompt: Null-terminated input prompt.
 * @ctx:    RlfCtx (result written to ctx->prompt_pp and ctx->env).
 */
void rlf_preprocess(const char *prompt, RlfCtx *ctx);

/**
 * rlf_concept_perceptron - Map prompt mean hidden → prefix tokens.
 *
 * @w:    Loaded RlfWeights.
 * @ctx:  RlfCtx (reads x_mean, writes prefix).
 */
void rlf_concept_perceptron(const RlfWeights *w, RlfCtx *ctx);

/**
 * rlf_bridge_encode - Project hidden state down to latent vector.
 *
 * @w:    Loaded RlfWeights.
 * @x:    Input [D_MODEL].
 * @ctx:  RlfCtx (writes bridge_latent).
 */
void rlf_bridge_encode(const RlfWeights *w, const rlf_f32 *x, RlfCtx *ctx);

/**
 * rlf_bridge_decode - Project latent vector back to D_MODEL.
 *
 * @w:    Loaded RlfWeights.
 * @ctx:  RlfCtx (reads bridge_latent, result returned in x_out).
 * @x_out: Output [D_MODEL] — caller-allocated.
 */
void rlf_bridge_decode(const RlfWeights *w, RlfCtx *ctx, rlf_f32 *x_out);

/**
 * rlf_infer - Full RLF reasoning chain.
 *
 * Runs the preprocessor, ConceptPerceptron, and up to RLF_MAX_LOOPS
 * iterations of Mamba backbone inference + bridge + lm_head argmax.
 * Stops when HALT token is produced or max loops reached.
 *
 * @prompt:    Null-terminated input prompt.
 * @backbone:  Pointer to loaded GGUF backbone (opaque to this header).
 * @weights:   Loaded OOSS sidecar weights.
 * @ctx:       Caller-allocated scratch context.
 * @answer:    Output null-terminated answer string buffer.
 * @answer_len: Capacity of answer buffer.
 * @n_loops:   If non-NULL, set to number of loops executed.
 *
 * Returns 0 on success (HALT reached), 1 if max loops hit, negative on error.
 */
int rlf_infer(
    const char     *prompt,
    void           *backbone,
    const RlfWeights *weights,
    RlfCtx         *ctx,
    char           *answer,
    int             answer_len,
    int            *n_loops
);

/**
 * rlf_status_print - Print loaded RLF config to REPL output.
 *
 * @w:  Loaded weights (or NULL if not yet loaded).
 */
void rlf_status_print(const RlfWeights *w);

/** rlf_ftoa - Convert float to string (no libc). buf must be >= 24 bytes. */
void rlf_ftoa(rlf_f32 v, char *buf);

#endif /* RLF_ENGINE_H */
