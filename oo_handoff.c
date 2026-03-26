/*
 * oo_handoff.c — Sovereign Handoff Protocol Implementation
 *
 * Handles OOHANDOFF.TXT receipt generation and sovereign_export.json parsing.
 * SHA-256 implementation is a minimal bare-metal version (no OpenSSL).
 */

#include "oo_handoff.h"
#include <string.h>
#include <stdio.h>

/* ── Minimal SHA-256 (FIPS 180-4 compliant, no external deps) ─────────────── */

static const uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static uint32_t sha_rotr(uint32_t x, int n)
{
    /**
     * Right rotate for SHA-256.
     */
    return (x >> n) | (x << (32 - n));
}

static void sha256_block(uint32_t state[8], const uint8_t block[64])
{
    /**
     * Process one 512-bit SHA-256 block.
     */
    uint32_t w[64];
    int i;

    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i*4] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }
    for (i = 16; i < 64; i++) {
        uint32_t s0 = sha_rotr(w[i-15], 7) ^ sha_rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = sha_rotr(w[i-2], 17) ^ sha_rotr(w[i-2], 19)  ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (i = 0; i < 64; i++) {
        uint32_t S1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + sha256_k[i] + w[i];
        uint32_t S0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void oo_sha256_hex(const void *data, uint64_t len, char *out_hex)
{
    /**
     * Compute SHA-256 of data buffer.
     * Writes 64-char hex string + null terminator.
     */
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    const uint8_t *p = (const uint8_t *)data;
    uint64_t i;

    /* Process complete 64-byte blocks */
    for (i = 0; i + 64 <= len; i += 64) {
        sha256_block(state, p + i);
    }

    /* Final block with padding */
    uint8_t final_block[128];
    uint64_t remaining = len - i;
    memset(final_block, 0, 128);
    memcpy(final_block, p + i, remaining);
    final_block[remaining] = 0x80;

    int blocks = (remaining < 56) ? 1 : 2;
    uint64_t bit_len = len * 8;
    uint8_t *length_pos = final_block + (blocks * 64 - 8);
    for (int j = 7; j >= 0; j--) {
        length_pos[j] = (uint8_t)(bit_len & 0xff);
        bit_len >>= 8;
    }

    sha256_block(state, final_block);
    if (blocks == 2) {
        sha256_block(state, final_block + 64);
    }

    /* Convert to hex */
    const char hex[] = "0123456789abcdef";
    for (int j = 0; j < 8; j++) {
        out_hex[j*8+0] = hex[(state[j] >> 28) & 0xf];
        out_hex[j*8+1] = hex[(state[j] >> 24) & 0xf];
        out_hex[j*8+2] = hex[(state[j] >> 20) & 0xf];
        out_hex[j*8+3] = hex[(state[j] >> 16) & 0xf];
        out_hex[j*8+4] = hex[(state[j] >> 12) & 0xf];
        out_hex[j*8+5] = hex[(state[j] >> 8) & 0xf];
        out_hex[j*8+6] = hex[(state[j] >> 4) & 0xf];
        out_hex[j*8+7] = hex[(state[j]) & 0xf];
    }
    out_hex[64] = '\0';
}


/* ── Receipt Init ─────────────────────────────────────────────────────────── */

void oo_receipt_init(
    OoHandoffReceipt *receipt,
    const MambaConfig *cfg,
    const char *organism_id,
    const char *genesis_id)
{
    /**
     * Populate receipt from current SSM config state.
     */
    memset(receipt, 0, sizeof(*receipt));

    if (organism_id) {
        strncpy(receipt->organism_id, organism_id, sizeof(receipt->organism_id) - 1);
    }
    if (genesis_id) {
        strncpy(receipt->genesis_id, genesis_id, sizeof(receipt->genesis_id) - 1);
    }

    strncpy(receipt->mode, "normal", sizeof(receipt->mode) - 1);
    receipt->lifeline_enabled = cfg ? cfg->lifeline_enabled : 1;
    receipt->rlf_max_loops = cfg ? cfg->max_rlf_loops : 8;
    receipt->boot_count = 1;
    receipt->continuity_epoch = 0;
}


/* ── Receipt Serialize ────────────────────────────────────────────────────── */

int oo_receipt_serialize(
    const OoHandoffReceipt *receipt,
    char *buf,
    int buf_size)
{
    /**
     * Serialize receipt as OOHANDOFF.TXT format (key=value lines).
     * Compatible with oo-host's read_key_value_file() parser.
     */
    if (!receipt || !buf || buf_size < 256) return -1;

    int n = snprintf(buf, buf_size,
        "organism_id=%s\n"
        "genesis_id=%s\n"
        "mode=%s\n"
        "continuity_epoch=%llu\n"
        "boot_count=%llu\n"
        "model_hash=%s\n"
        "last_inference_ts=%llu\n"
        "lifeline_enabled=%d\n"
        "rlf_max_loops=%d\n"
        "quant_type=%d\n"
        "model_name=%s\n",
        receipt->organism_id,
        receipt->genesis_id,
        receipt->mode,
        (unsigned long long)receipt->continuity_epoch,
        (unsigned long long)receipt->boot_count,
        receipt->model_hash,
        (unsigned long long)receipt->last_inference_ts,
        receipt->lifeline_enabled,
        receipt->rlf_max_loops,
        receipt->quant_type,
        receipt->model_name
    );

    return (n > 0 && n < buf_size) ? n : -1;
}


/* ── Host Export Parser ───────────────────────────────────────────────────── */

/**
 * Extract a string value for a JSON key (minimal JSON parser).
 * Returns length of value, or -1 if not found.
 */
static int json_extract_str(const char *json, int json_len,
                            const char *key, char *out, int out_size)
{
    /**
     * Simple key:"value" extractor — no nested objects.
     */
    char pattern[128];
    int klen = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (klen < 0) return -1;

    const char *found = NULL;
    for (int i = 0; i <= json_len - klen; i++) {
        if (memcmp(json + i, pattern, klen) == 0) {
            found = json + i + klen;
            break;
        }
    }
    if (!found) return -1;

    /* Skip to colon, then to opening quote */
    while (*found && *found != ':') found++;
    if (!*found) return -1;
    found++;
    while (*found == ' ' || *found == '\t') found++;

    if (*found == '"') {
        /* String value */
        found++;
        int len = 0;
        while (found[len] && found[len] != '"' && len < out_size - 1) {
            out[len] = found[len];
            len++;
        }
        out[len] = '\0';
        return len;
    }

    return -1;
}

/**
 * Extract an integer value for a JSON key.
 */
static int json_extract_int(const char *json, int json_len,
                            const char *key, int *out)
{
    /**
     * Simple key:number extractor.
     */
    char pattern[128];
    int klen = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (klen < 0) return -1;

    const char *found = NULL;
    for (int i = 0; i <= json_len - klen; i++) {
        if (memcmp(json + i, pattern, klen) == 0) {
            found = json + i + klen;
            break;
        }
    }
    if (!found) return -1;

    while (*found && *found != ':') found++;
    if (!*found) return -1;
    found++;
    while (*found == ' ' || *found == '\t') found++;

    /* Parse number (possibly true/false) */
    if (*found == 't') { *out = 1; return 0; }
    if (*found == 'f') { *out = 0; return 0; }

    int val = 0, neg = 0;
    if (*found == '-') { neg = 1; found++; }
    while (*found >= '0' && *found <= '9') {
        val = val * 10 + (*found - '0');
        found++;
    }
    *out = neg ? -val : val;
    return 0;
}

int oo_host_export_parse(
    OoHostExport *export,
    const char *json_buf,
    int json_len)
{
    /**
     * Parse sovereign_export.json into OoHostExport struct.
     * Extracts top-level fields and first goal from top_goals array.
     */
    if (!export || !json_buf || json_len <= 0) return -1;
    memset(export, 0, sizeof(*export));

    json_extract_str(json_buf, json_len, "organism_id",
                     export->organism_id, sizeof(export->organism_id));
    json_extract_str(json_buf, json_len, "mode",
                     export->mode, sizeof(export->mode));

    int val = 0;
    if (json_extract_int(json_buf, json_len, "continuity_epoch", &val) == 0)
        export->continuity_epoch = (uint64_t)val;
    if (json_extract_int(json_buf, json_len, "boot_or_start_count", &val) == 0)
        export->boot_count = (uint64_t)val;
    if (json_extract_int(json_buf, json_len, "active_goal_count", &val) == 0)
        export->active_goal_count = val;

    /* Policy fields */
    json_extract_int(json_buf, json_len, "safe_first", &export->policy_safe_first);
    json_extract_int(json_buf, json_len, "deny_by_default", &export->policy_deny_default);
    json_extract_str(json_buf, json_len, "enforcement",
                     export->policy_enforcement, sizeof(export->policy_enforcement));

    /* First goal title/status (simple extraction) */
    json_extract_str(json_buf, json_len, "title",
                     export->top_goal_title, sizeof(export->top_goal_title));
    json_extract_str(json_buf, json_len, "status",
                     export->top_goal_status, sizeof(export->top_goal_status));

    return 0;
}
