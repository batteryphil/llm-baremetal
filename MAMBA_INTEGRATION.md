# Mamba2 Baremetal Integration — Research Notes

> Integration attempt: **Mamba-2 2.7B RLF Engine → llm-baremetal UEFI runtime**
> Status: **PARTIALLY BLOCKED** (AVX2 `#UD` in UEFI firmware)

This document records what was built, what worked, and what is blocked when
attempting to run the Mamba-2 2.7B Recursive Latent Forcing (RLF) inference
engine inside this UEFI bare-metal runtime.

Related repo: [batteryphil/mamba2backbonerecursion](https://github.com/batteryphil/mamba2backbonerecursion)

---

## What Was Built

### New Files Added to This Repo

| File | Description |
|------|-------------|
| `ssm_infer.c` / `ssm_infer.h` | SSM (State Space Model) inference kernel — forward pass for a single Mamba layer in C |
| `ssm_infer_avx2.c` | AVX2-accelerated variant of the SSM forward pass (256-bit SIMD) |
| `ssm_weights.c` / `ssm_weights.h` | Weight loader for the exported Mamba binary weight format |
| `bpe_tokenizer.c` / `bpe_tokenizer.h` | BPE tokenizer for Mamba's `EleutherAI/gpt-neox-20b` tokenizer |
| `llama2_efi_mamba.c` | UEFI EFI application wrapping the SSM inference kernel |
| `export_mamba_baremetal.py` | Python script to export Mamba-2 2.7B weights → flat binary format |
| `export_bpe_table.py` | Export the GPT-NeoX BPE tokenizer → binary table for C runtime |

### Python-Side Export

`export_mamba_baremetal.py` successfully exports:
- All 64 Mamba-2 layers (d_model=2560, d_state=128, d_conv=4, expand=2)
- LoRA adapters for layers 48-63
- Prefix latent scratchpad (8 × 2560)
- Latent bridge weights (2560→64→2560)
- Lifeline gate vector (2560)
- LM head projection (50280 × 2560)

Output: `model.mamba.bin` (2.85 GB flat binary, bfloat16)

The weight export **works correctly** and loads into the C runtime.

---

## What Works

- ✅ Weight export (`export_mamba_baremetal.py`) — produces valid `model.mamba.bin`
- ✅ BPE tokenizer export (`export_bpe_table.py`) — `tokenizer.bin` works in C
- ✅ `ssm_weights.c` — loads the 2.85 GB binary correctly in QEMU with 8GB RAM
- ✅ `bpe_tokenizer.c` — tokenizes input correctly, matches Python reference
- ✅ `llama2_efi_mamba.c` — compiles and links against GNU-EFI, boots in QEMU
- ✅ QEMU boots the `.efi` image and reaches the REPL initialization code

---

## What Is Blocked

### AVX2 `#UD` Exception in UEFI

**Problem:** The SSM inference kernel (`ssm_infer.c`) uses AVX2 SIMD intrinsics
(`_mm256_*`) for the matrix-vector multiply inner loop. When this code executes
inside a UEFI pre-boot environment (even under QEMU with `accel=kvm`), it
triggers a `#UD` (Invalid Opcode) exception and the system faults.

**Root cause:** UEFI firmware runs in 64-bit protected mode but does **not**
initialize the XSAVE/AVX CPU state via `XSETBV`. The OS is responsible for
calling `XSETBV` to enable AVX/AVX2 register state (YMM registers) during
boot. Before this happens, any AVX2 instruction faults with `#UD`.

**Impact:**
- `ssm_infer_avx2.c` is completely unusable in UEFI
- The scalar fallback in `ssm_infer.c` works but is ~8-12× slower
- At scalar speed, a single Mamba-2 forward pass through 64 layers takes
  ~45 seconds on QEMU (Intel i7, host-speed), making interactive inference
  impractical

**Attempted workaround:** Stripped all AVX2 objects from the build
(`make NOAVX2=1`). The resulting `.efi` links and boots, but inference
is too slow to be useful.

---

## Path Forward

### Option A — XSETBV in EFI Application (Medium effort)

Call `XSETBV` with `XCR0 |= 0x6` (`XSTATE_SSE | XSTATE_YMM`) before executing
any AVX2 code. This requires:
1. Reading `XCR0` via `XGETBV`
2. Setting bits 1-2 (`SSE state` + `AVX/YMM state`)
3. Writing back with `XSETBV`

This is architecturally correct but requires inline assembly in the EFI app
and careful ordering (must happen before any SSM layer call).

### Option B — Pure Scalar SSM Kernel (Low effort, usable)

Replace `ssm_infer_avx2.c` with a clean scalar implementation using loop
unrolling and compiler auto-vectorization hints (`__builtin_expect`, `restrict`
pointers). With `-O3 -march=native` the compiler can often generate SSE2 code
(which IS valid in UEFI without `XSETBV`) and recover ~3-4× over naive scalar.

At SSE2 speed (estimated ~12s per forward pass), a 1-hop chain query is
borderline usable for the factual offloading use-case.

### Option C — Quantize to INT8 (Higher effort, best result)

Quantize the 2.7B model to INT8 using the existing `djiblas.c` quantization
routines. INT8 GEMV uses only SSE2 (always available in UEFI) and reduces the
model from 2.85GB → ~1.4GB, making it fit with headroom in the 8GB FAT image.

---

## Memory Profile

| Component | Size |
|-----------|------|
| Mamba-2 2.7B weights (bfloat16) | 2.85 GB |
| SSM state buffers (64 layers × 128 states × 2560) | ~105 MB |
| KV-equivalent (none — SSM is O(1)) | 0 MB |
| Tokenizer table | 27 MB |
| **Total runtime RAM** | **~3.0 GB** |

Fits in 8GB systems with the current UEFI heap allocator.

---

## Build

```bash
# Build without AVX2 (scalar fallback, UEFI-safe)
make NOAVX2=1 MAMBA=1

# Build the full Mamba EFI application
make llama2_efi_mamba.efi
```

The `MAMBA=1` flag switches the inference backend from the GGUF `llama2.c`
path to the `ssm_infer.c` / `ssm_weights.c` Mamba path.

---

## Credits

- **[Djiby Diop](https://github.com/Djiby-diop)** — UEFI runtime, DjiBLAS, 
  GGUF loader, heap allocator, BPE tokenizer base. The SSM kernel (`ssm_infer.c`)
  was implemented as an extension of his DjiBLAS math layer following his
  established coding patterns.
- **[State Spaces / Mamba](https://github.com/state-spaces/mamba)** — Mamba-2
  architecture.
