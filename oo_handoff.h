/*
 * oo_handoff.h — Sovereign Handoff Protocol for Operating Organism
 *
 * Writes OOHANDOFF.TXT receipt to boot volume for oo-host sync checking.
 * Reads sovereign_export.json from host for goals/policy awareness.
 *
 * Protocol flow:
 *   oo-host (Rust)  → sovereign_export.json → sovereign (UEFI/C)
 *   sovereign (UEFI/C) → OOHANDOFF.TXT      → oo-host (Rust)
 */

#ifndef OO_HANDOFF_H
#define OO_HANDOFF_H

#include "ssm_infer.h"
#include <stdint.h>

/* ── Handoff Receipt (written by sovereign) ───────────────────────────────── */

typedef struct {
    char     organism_id[64];        /* UUID from identity                  */
    char     genesis_id[64];         /* Genesis UUID                        */
    char     mode[16];               /* "normal", "degraded", "safe"        */
    uint64_t continuity_epoch;       /* Monotonic epoch counter             */
    uint64_t boot_count;             /* Boot/start counter                  */
    char     model_hash[65];         /* SHA-256 hex of .mamba.bin (64+null) */
    uint64_t last_inference_ts;      /* Epoch seconds of last inference     */
    int      lifeline_enabled;       /* 1=on, 0=off                         */
    int      rlf_max_loops;          /* Current RLF loop depth              */
    int      quant_type;             /* 0=fp32, 1=int8                      */
    char     model_name[64];         /* Model filename                      */
} OoHandoffReceipt;

/* ── Host Export (read by sovereign) ──────────────────────────────────────── */

typedef struct {
    char     organism_id[64];
    char     mode[16];
    uint64_t continuity_epoch;
    uint64_t boot_count;

    /* Policy (from oo-host) */
    int      policy_safe_first;      /* 1=safe-first policy active          */
    int      policy_deny_default;    /* 1=deny-by-default                   */
    char     policy_enforcement[16]; /* "off", "observe", "enforce"         */

    /* Goals summary */
    int      active_goal_count;
    char     top_goal_title[128];    /* Title of highest-priority goal      */
    char     top_goal_status[16];    /* "pending", "doing", "blocked"       */
} OoHostExport;

/* ── API ─────────────────────────────────────────────────────────────────── */

/**
 * Initialize a handoff receipt with current SSM state.
 */
void oo_receipt_init(
    OoHandoffReceipt *receipt,
    const MambaConfig *cfg,
    const char *organism_id,
    const char *genesis_id
);

/**
 * Serialize receipt to OOHANDOFF.TXT format (key=value lines).
 * Returns number of bytes written, or -1 on error.
 */
int oo_receipt_serialize(
    const OoHandoffReceipt *receipt,
    char *buf,
    int buf_size
);

/**
 * Parse a sovereign_export.json buffer into OoHostExport struct.
 * Returns 0 on success, -1 on error.
 */
int oo_host_export_parse(
    OoHostExport *export,
    const char *json_buf,
    int json_len
);

/**
 * Compute SHA-256 hash of a data buffer (model file).
 * Writes 64-char hex string + null to out_hex (must be ≥65 bytes).
 */
void oo_sha256_hex(const void *data, uint64_t len, char *out_hex);

#endif /* OO_HANDOFF_H */
