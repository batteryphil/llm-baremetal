#!/bin/bash
# ─────────────────────────────────────────────────────────────
# test-qemu-handoff.sh — Full QEMU UEFI test pipeline
#
# This script:
#   1. Exports the trained model to .mamba.bin (INT8 or FP32)
#   2. Exports the BPE tokenizer
#   3. Builds the UEFI binary
#   4. Creates a bootable FAT32 disk image
#   5. Copies model + tokenizer + OOHANDOFF.TXT to the disk
#   6. Launches QEMU with OVMF firmware
#
# Prerequisites:
#   sudo apt install qemu-system-x86 ovmf gnu-efi mtools dosfstools
#
# Usage:
#   ./test-qemu-handoff.sh              # Full pipeline
#   ./test-qemu-handoff.sh --int8       # With INT8 quantization
#   ./test-qemu-handoff.sh --skip-build # Skip build, just launch
#   ./test-qemu-handoff.sh --nokvm      # No KVM (WSL/container)
# ─────────────────────────────────────────────────────────────

set -e

QUANTIZE="fp32"
SKIP_BUILD=0
NO_KVM=0

for arg in "$@"; do
    case "$arg" in
        --int8)       QUANTIZE="int8" ;;
        --skip-build) SKIP_BUILD=1 ;;
        --nokvm)      NO_KVM=1 ;;
    esac
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Mamba2 RLF — QEMU UEFI Handoff Test"
echo "  Quantize: $QUANTIZE  |  KVM: $( [ $NO_KVM -eq 0 ] && echo 'yes' || echo 'no' )"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Check prerequisites ────────────────────────────────────────
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: $1 not found. Install with: $2"
        exit 1
    fi
}

check_cmd qemu-system-x86_64 "sudo apt install qemu-system-x86"
check_cmd mkfs.fat           "sudo apt install dosfstools"
check_cmd mmd                "sudo apt install mtools"

# Find OVMF firmware
OVMF=""
for path in /usr/share/OVMF/OVMF_CODE.fd /usr/share/OVMF/OVMF_CODE_4M.fd /usr/share/qemu/OVMF.fd; do
    if [ -f "$path" ]; then
        OVMF="$path"
        break
    fi
done
if [ -z "$OVMF" ]; then
    echo "ERROR: OVMF not found. Install with: sudo apt install ovmf"
    exit 1
fi
echo "  OVMF: $OVMF"

if [ $SKIP_BUILD -eq 0 ]; then
    # ── Step 1: Export model ───────────────────────────────────────
    echo ""
    echo "── Step 1: Export model ($QUANTIZE) ──"
    if [ "$QUANTIZE" = "int8" ]; then
        python export_mamba_baremetal.py --quantize int8
    else
        python export_mamba_baremetal.py
    fi

    # ── Step 2: Export tokenizer ───────────────────────────────────
    echo ""
    echo "── Step 2: Export BPE tokenizer ──"
    python export_bpe_table.py

    # ── Step 3: Build UEFI binary ──────────────────────────────────
    echo ""
    echo "── Step 3: Build UEFI binary ──"
    make clean
    make
fi

# ── Step 4: Create boot disk ──────────────────────────────────
echo ""
echo "── Step 4: Create bootable disk ──"
make disk-model

# ── Step 5: Launch QEMU ───────────────────────────────────────
echo ""
echo "── Step 5: Launching QEMU ──"
echo ""
echo "  REPL commands available:"
echo "    /help       — Show all commands"
echo "    /ssm_info   — Model info"
echo "    /telemetry  — OO engine metrics"
echo "    /handoff write — Generate OOHANDOFF.TXT"
echo "    Ctrl-A X    — Exit QEMU"
echo ""

if [ $NO_KVM -eq 0 ]; then
    make qemu
else
    make qemu-nokvm
fi
