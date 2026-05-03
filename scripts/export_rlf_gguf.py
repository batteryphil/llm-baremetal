#!/usr/bin/env python3
"""
export_rlf_gguf.py — Export Mamba-1.4B RLF checkpoint to GGUF + OOSS sidecar.
==============================================================================
Produces:
  <out>.gguf   Backbone + LoRA merged + our lm_head (Q8_0 quantised)
  <out>.ooss   RLF sidecar: ConceptPerceptron, bridge, gate, norm weights (F16)

Usage:
    python scripts/export_rlf_gguf.py \
        --ckpt /hdd_data/rlf-1.4b-checkpoints/final \
        --base state-spaces/mamba-1.4b \
        --out models/mamba1_4b_rlf

The resulting files are suitable for the llm-baremetal bare-metal runtime.
Load with: /core_load mamba1_4b_rlf.gguf  then  /rlf_load mamba1_4b_rlf.ooss
"""

import argparse
import struct
import sys
from pathlib import Path

import torch
import numpy as np

# mamba_ssm must be importable — add project dir to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       '../mamba2backbonerecursion/mamba14b'))
from mamba_ssm import MambaLMHeadModel


# ── Constants ────────────────────────────────────────────────────────────────

GGUF_MAGIC   = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGUF value types
GGUF_UINT32 = 4
GGUF_INT32  = 5
GGUF_FLOAT32 = 6
GGUF_STRING  = 8
GGUF_UINT16  = 7

# GGUF tensor types
GGML_F32  = 0
GGML_F16  = 1
GGML_Q8_0 = 8

HALT_TOKEN_ID = 7803   # § token in GPT-NeoX vocab
MAX_LOOPS     = 6
PREFIX_M      = 8
D_MODEL       = 2048
D_BRIDGE      = 128
LORA_SPLIT    = 24     # layers 0-23 frozen, 24-47 had LoRA


# ── Q8_0 quantisation ────────────────────────────────────────────────────────

def quantise_q8_0(tensor: np.ndarray) -> bytes:
    """Quantise a float32 tensor to GGML Q8_0 format.

    Q8_0 block layout (32 elements per block):
      [float16 scale][32 × int8 values]  = 2 + 32 = 34 bytes per block
    """
    t = tensor.astype(np.float32).flatten()
    n = len(t)
    pad = (32 - n % 32) % 32
    if pad:
        t = np.concatenate([t, np.zeros(pad, dtype=np.float32)])

    blocks = t.reshape(-1, 32)
    out = bytearray()
    for blk in blocks:
        amax = np.max(np.abs(blk))
        scale = amax / 127.0 if amax > 0 else 1.0
        quant = np.round(blk / scale).astype(np.int8)
        scale_f16 = np.float16(scale)
        out += struct.pack('<e', float(scale_f16))  # 2 bytes
        out += quant.tobytes()                       # 32 bytes
    return bytes(out)


def quantise_f16(tensor: np.ndarray) -> bytes:
    """Cast tensor to float16 and return raw bytes."""
    return tensor.astype(np.float16).tobytes()


# ── GGUF writer ──────────────────────────────────────────────────────────────

class GgufWriter:
    """Minimal GGUF v3 file builder."""

    def __init__(self) -> None:
        """Initialise empty writer."""
        self._meta: list[tuple[str, int, object]] = []
        self._tensors: list[tuple[str, tuple, int, bytes]] = []

    def add_string(self, key: str, value: str) -> None:
        """Add a string metadata key."""
        self._meta.append((key, GGUF_STRING, value))

    def add_uint32(self, key: str, value: int) -> None:
        """Add a uint32 metadata key."""
        self._meta.append((key, GGUF_UINT32, value))

    def add_float32(self, key: str, value: float) -> None:
        """Add a float32 metadata key."""
        self._meta.append((key, GGUF_FLOAT32, value))

    def add_tensor(self, name: str, shape: tuple,
                   dtype: int, data: bytes) -> None:
        """Register a tensor to write."""
        self._tensors.append((name, shape, dtype, data))

    def _encode_string(self, s: str) -> bytes:
        """Encode a GGUF length-prefixed string."""
        enc = s.encode('utf-8')
        return struct.pack('<Q', len(enc)) + enc

    def _encode_meta_value(self, vtype: int, value: object) -> bytes:
        """Encode a single metadata value."""
        if vtype == GGUF_STRING:
            return self._encode_string(value)
        if vtype == GGUF_UINT32:
            return struct.pack('<I', value)
        if vtype == GGUF_FLOAT32:
            return struct.pack('<f', value)
        raise ValueError(f"Unsupported GGUF type {vtype}")

    def write(self, path: Path) -> None:
        """Write the GGUF file."""
        # Build tensor info section first so we know data offsets
        tensor_info = b''
        tensor_data = b''
        data_offset = 0

        for name, shape, dtype, data in self._tensors:
            # Align to 32 bytes
            align = (32 - len(tensor_data) % 32) % 32
            tensor_data += b'\x00' * align
            data_offset = len(tensor_data)

            name_enc   = self._encode_string(name)
            n_dims     = len(shape)
            dims_bytes = struct.pack(f'<I{"Q"*n_dims}', n_dims, *shape)
            dtype_off  = struct.pack('<IQ', dtype, data_offset)

            tensor_info += name_enc + dims_bytes + dtype_off
            tensor_data += data

        # Build metadata section
        meta_bytes = b''
        for key, vtype, value in self._meta:
            meta_bytes += self._encode_string(key)
            meta_bytes += struct.pack('<I', vtype)
            meta_bytes += self._encode_meta_value(vtype, value)

        # Header: magic, version, n_tensors, n_kv
        header = struct.pack('<IIQQ',
                             GGUF_MAGIC, GGUF_VERSION,
                             len(self._tensors), len(self._meta))

        with open(path, 'wb') as f:
            f.write(header)
            f.write(meta_bytes)
            f.write(tensor_info)
            # Align data section to 32 bytes from file start
            pos = f.tell()
            pad = (32 - pos % 32) % 32
            f.write(b'\x00' * pad)
            f.write(tensor_data)

        print(f'  GGUF written: {path} ({path.stat().st_size / 1e6:.1f} MB)')


# ── OOSS sidecar writer ───────────────────────────────────────────────────────

def write_ooss(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write RLF OOSS sidecar in simple header+tensor format.

    Format:
      [magic 4B "OOSS"][version 4B = 2][n_tensors 4B]
      For each tensor:
        [name_len 4B][name bytes][n_dims 4B][shape: n_dims × 8B uint64]
        [dtype 4B: 0=F32, 1=F16][data_len 8B][data bytes]
    """
    MAGIC   = b'OOSS'
    VERSION = 2

    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<II', VERSION, len(tensors)))
        for name, arr in tensors.items():
            arr_f16   = arr.astype(np.float16)
            data      = arr_f16.tobytes()
            name_enc  = name.encode('utf-8')
            shape     = arr.shape
            f.write(struct.pack('<I', len(name_enc)) + name_enc)
            f.write(struct.pack(f'<I{"Q"*len(shape)}', len(shape), *shape))
            f.write(struct.pack('<IQ', 1, len(data)))  # dtype=1 (F16)
            f.write(data)

    print(f'  OOSS written: {path} ({path.stat().st_size / 1e6:.1f} MB)')


# ── LoRA merge ───────────────────────────────────────────────────────────────

def merge_lora(base_state: dict, lora_state: dict,
               lora_split: int = LORA_SPLIT) -> dict:
    """Merge LoRA adapter weights into the base model state dict.

    LoRA adds delta = scale * (B @ A) to layers >= lora_split.
    The merged weights are returned; the base_state is not mutated.

    Args:
        base_state: Base model state dict (will be copied for merge).
        lora_state: LoRA state dict from lora.pt.
        lora_split: First layer index that had LoRA applied.

    Returns:
        Merged state dict with LoRA baked in.
    """
    merged = {k: v.clone() for k, v in base_state.items()}

    # LoRA format from our training: keys are like
    #   "layers.N.mixer.in_proj.lora_A" / "lora_B" with optional "lora_scale"
    lora_A: dict[str, torch.Tensor] = {}
    lora_B: dict[str, torch.Tensor] = {}
    lora_scale: dict[str, float]    = {}

    for k, v in lora_state.items():
        if '.lora_A' in k:
            base_key = k.replace('.lora_A', '')
            lora_A[base_key] = v
        elif '.lora_B' in k:
            base_key = k.replace('.lora_B', '')
            lora_B[base_key] = v
        elif '.lora_scale' in k:
            base_key = k.replace('.lora_scale', '')
            lora_scale[base_key] = v.item()

    applied = 0
    for key in lora_A:
        if key not in lora_B:
            continue
        # LoRA keys are bare 'layers.N...' but base state dict has 'backbone.layers.N...'
        base_key = 'backbone.' + key
        if base_key not in merged:
            print(f'  WARN: LoRA key {key} not in base model, skipping')
            continue
        A     = lora_A[key].float()      # [r, d_in]
        B     = lora_B[key].float()      # [d_out, r]
        scale = lora_scale.get(key, 1.0)
        delta = scale * (B @ A)          # [d_out, d_in]
        merged[base_key] = merged[base_key].float() + delta.to(merged[base_key].dtype)
        applied += 1

    print(f'  LoRA: merged {applied} adapter pairs into base weights')
    return merged


# ── Main export ──────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full export pipeline."""
    parser = argparse.ArgumentParser(
        description='Export Mamba-1.4B RLF checkpoint to GGUF + OOSS'
    )
    parser.add_argument('--ckpt', required=True,
                        help='Path to RLF checkpoint directory (final/)')
    parser.add_argument('--base', default='state-spaces/mamba-1.4b',
                        help='HuggingFace model ID or local path for base model')
    parser.add_argument('--out', required=True,
                        help='Output path prefix (no extension)')
    parser.add_argument('--quant', default='q8_0',
                        choices=['f16', 'q8_0'],
                        help='Quantisation for backbone tensors')
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt)
    out_gguf = Path(args.out).with_suffix('.gguf')
    out_ooss = Path(args.out).with_suffix('.ooss')
    out_gguf.parent.mkdir(parents=True, exist_ok=True)

    print('=== RLF GGUF Export ===')

    # 1. Load base model using MambaLMHeadModel (matches training setup)
    print(f'\n[1/5] Loading base model: {args.base}')
    base_model = MambaLMHeadModel.from_pretrained(
        args.base, dtype=torch.float32, device='cpu'
    )
    base_state = base_model.state_dict()
    print(f'  Base params: {sum(p.numel() for p in base_model.parameters()):,}')

    # 2. Merge LoRA
    print(f'\n[2/5] Merging LoRA from {ckpt_dir}/lora.pt')
    lora_path = ckpt_dir / 'lora.pt'
    if lora_path.exists():
        lora_state = torch.load(
            lora_path, map_location='cpu', weights_only=True
        )
        merged_state = merge_lora(base_state, lora_state)
    else:
        print('  No lora.pt found — using unmodified base weights')
        merged_state = base_state

    # 3. Load our trained lm_head
    print(f'\n[3/5] Loading trained lm_head')
    lm_head_state = torch.load(
        ckpt_dir / 'lm_head.pt', map_location='cpu', weights_only=True
    )
    # lm_head.pt stores state dict of nn.Linear
    lm_head_w = lm_head_state.get('weight',
                 lm_head_state.get('lm_head.weight',
                 next(iter(lm_head_state.values()))))
    print(f'  lm_head shape: {tuple(lm_head_w.shape)}')

    # 4. Build GGUF
    print(f'\n[4/5] Building GGUF ({args.quant})')
    writer = GgufWriter()

    # Metadata
    writer.add_string('general.architecture',   'mamba')
    writer.add_string('general.name',           'mamba-1.4b-rlf-v5')
    writer.add_uint32('mamba.context_length',    2048)
    writer.add_uint32('mamba.embedding_length',  2048)
    writer.add_uint32('mamba.feed_forward_length', 4096)
    writer.add_uint32('mamba.block_count',       48)
    writer.add_uint32('mamba.ssm.conv_kernel',   4)
    writer.add_uint32('mamba.ssm.inner_size',    4096)
    writer.add_uint32('mamba.ssm.state_size',    16)
    writer.add_uint32('tokenizer.ggml.model',    0)
    # RLF-specific metadata (read by rlf_engine.c)
    writer.add_uint32('rlf.halt_token_id',  HALT_TOKEN_ID)
    writer.add_uint32('rlf.max_loops',      MAX_LOOPS)
    writer.add_uint32('rlf.prefix_m',       PREFIX_M)
    writer.add_uint32('rlf.d_model',        D_MODEL)
    writer.add_uint32('rlf.d_bridge',       D_BRIDGE)
    writer.add_uint32('rlf.lora_split',     LORA_SPLIT)

    quantise = quantise_q8_0 if args.quant == 'q8_0' else quantise_f16
    dtype_id  = GGML_Q8_0   if args.quant == 'q8_0' else GGML_F16

    # Map HF Mamba state dict keys → GGUF tensor names
    # HF key pattern: backbone.layers.N.mixer.{in_proj,out_proj,x_proj,...}.weight
    def hf_to_gguf_name(k: str) -> str | None:
        """Convert HF Mamba state dict key to GGUF tensor name."""
        k = k.replace('backbone.', '')
        if k == 'embedding.weight':
            return 'token_embd.weight'
        if k == 'norm_f.weight':
            return 'output_norm.weight'
        if not k.startswith('layers.'):
            return None
        parts = k.split('.')
        # parts: ['layers', 'N', component, ...]
        layer = parts[1]
        rest  = '.'.join(parts[2:])
        mapping = {
            'mixer.in_proj.weight':  f'blk.{layer}.ssm_in.weight',
            'mixer.out_proj.weight': f'blk.{layer}.ssm_out.weight',
            'mixer.x_proj.weight':   f'blk.{layer}.ssm_x.weight',
            'mixer.dt_proj.weight':  f'blk.{layer}.ssm_dt.weight',
            'mixer.dt_proj.bias':    f'blk.{layer}.ssm_dt.bias',
            'mixer.A_log':           f'blk.{layer}.ssm_a',
            'mixer.D':               f'blk.{layer}.ssm_d',
            'mixer.conv1d.weight':   f'blk.{layer}.ssm_conv.weight',
            'mixer.conv1d.bias':     f'blk.{layer}.ssm_conv.bias',
            'norm.weight':           f'blk.{layer}.attn_norm.weight',
        }
        return mapping.get(rest)

    n_tensors = 0
    for hf_key, tensor in merged_state.items():
        gguf_name = hf_to_gguf_name(hf_key)
        if gguf_name is None:
            continue
        arr = tensor.float().numpy()
        # Keep norms and biases in F32, quantise the rest
        if 'norm' in gguf_name or 'bias' in gguf_name or arr.ndim == 1:
            data   = arr.astype(np.float32).tobytes()
            dt     = GGML_F32
        else:
            data   = quantise(arr)
            dt     = dtype_id
        writer.add_tensor(gguf_name, tuple(arr.shape), dt, data)
        n_tensors += 1

    # lm_head as output.weight
    lm_np  = lm_head_w.float().numpy()
    writer.add_tensor('output.weight', tuple(lm_np.shape),
                      dtype_id, quantise(lm_np))
    n_tensors += 1

    print(f'  Exporting {n_tensors} tensors...')
    writer.write(out_gguf)

    # 5. Build OOSS sidecar (RLF components — keep F16 for precision)
    print(f'\n[5/5] Building OOSS sidecar')
    ooss_tensors: dict[str, np.ndarray] = {}

    def load_pt(fname: str) -> dict:
        """Load a .pt state dict from the checkpoint directory."""
        return torch.load(ckpt_dir / fname, map_location='cpu',
                          weights_only=True)

    # ConceptPerceptron layers
    cp_state = load_pt('concept_perceptron.pt')
    for k, v in cp_state.items():
        # keys: 'scale', 'mapper.0.weight', 'mapper.0.bias', 'mapper.2.weight', 'mapper.2.bias'
        safe_key = 'rlf.perceptron.' + k.replace('.', '_')
        ooss_tensors[safe_key] = v.float().numpy()
        print(f'  + {safe_key}: {tuple(v.shape)}')

    # Latent bridge
    bd_state = load_pt('bridge_down.pt')
    bu_state = load_pt('bridge_up.pt')
    for k, v in bd_state.items():
        ooss_tensors['rlf.bridge_down.' + k.replace('.', '_')] = v.float().numpy()
    for k, v in bu_state.items():
        ooss_tensors['rlf.bridge_up.' + k.replace('.', '_')] = v.float().numpy()

    # Lifeline gate — stored as raw tensor [D_MODEL], not a state dict
    lg_raw = load_pt('lifeline_gate.pt')
    if isinstance(lg_raw, dict):
        for k, v in lg_raw.items():
            ooss_tensors['rlf.lifeline.' + k.replace('.', '_')] = v.float().numpy()
    else:
        ooss_tensors['rlf.lifeline.vector'] = lg_raw.float().numpy()
        print(f'  + rlf.lifeline.vector: {tuple(lg_raw.shape)}')

    # Loop RMSNorm
    ln_state = load_pt('loop_norm.pt')
    for k, v in ln_state.items():
        ooss_tensors['rlf.loop_norm.' + k.replace('.', '_')] = v.float().numpy()
        print(f'  + rlf.loop_norm.{k}: {tuple(v.shape)}')

    write_ooss(out_ooss, ooss_tensors)

    # Summary
    print('\n=== Export complete ===')
    print(f'  GGUF: {out_gguf} ({out_gguf.stat().st_size / 1e6:.0f} MB)')
    print(f'  OOSS: {out_ooss} ({out_ooss.stat().st_size / 1e6:.0f} MB)')
    print(f'\nLoad in llm-baremetal:')
    print(f'  /core_load {out_gguf.name}')
    print(f'  /rlf_load  {out_ooss.name}')


if __name__ == '__main__':
    main()
