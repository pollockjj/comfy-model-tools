#!/usr/bin/env python3
"""Extract the Gemma-4 MTP (multi-token-prediction) drafter head from the llama.cpp
GGUF drafter into a ComfyUI safetensors sidecar for self-speculative decode.

The MTP head is NOT in google/gemma-4-E{2,4}B-it/model.safetensors (header-scanned:
zero mtp/nextn keys) — its only public source is the llama.cpp `gemma4-assistant`
GGUF (unsloth `MTP/mtp-gemma-4-E{2,4}B-it-Q8_0.gguf`). This job dequantizes those
Q8_0 tensors and remaps GGUF names -> ComfyUI module keys.

Head architecture (GGUF-confirmed): 4 layers @ hidden 256, head_count 4 / kv 2,
head_dim 64, ffn 2048 GELU-par; nextn.pre_projection [5120->256] consumes
concat(target_tok_embd 2560, backbone_hidden 2560); nextn.post_projection [256->2560]
feeds the shared backbone lm_head. Small (256-dim) layers stay bf16 — int8 would not
pay at those shapes; the caller int8-convrots the two 2560-wide projections separately.

    python convert_gemma4_mtp.py --gguf mtp-gemma-4-E4B-it-Q8_0.gguf \
        --out gemma4_e4b_it_mtp.safetensors

Matmul-free (bf16 dequant + name remap), so byte-identical on any machine.
"""
import argparse
import re
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, "/home/johnj/dev_master/llama.cpp-master/gguf-py")
from gguf import GGUFReader, dequantize  # noqa: E402

# GGUF blk.* name -> ComfyUI decoder-layer suffix
BLK_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    "layer_output_scale.weight": "layer_scalar",
}
TOP_MAP = {
    "nextn.pre_projection.weight": "mtp.pre_projection.weight",
    "nextn.post_projection.weight": "mtp.post_projection.weight",
    "output_norm.weight": "mtp.norm.weight",
    "token_embd.weight": "mtp.embed_tokens.weight",
    "rope_freqs.weight": "mtp.rope_freqs.weight",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = GGUFReader(args.gguf)
    out = {}
    for t in r.tensors:
        w = torch.from_numpy(dequantize(t.data, t.tensor_type).copy()).to(torch.bfloat16)
        m = re.match(r"blk\.(\d+)\.(.+)", t.name)
        if m:
            idx, suffix = m.group(1), m.group(2)
            if suffix not in BLK_MAP:
                raise SystemExit(f"unmapped blk tensor: {t.name}")
            key = f"mtp.layers.{idx}.{BLK_MAP[suffix]}"
        elif t.name in TOP_MAP:
            key = TOP_MAP[t.name]
        else:
            raise SystemExit(f"unmapped tensor: {t.name}")
        out[key] = w.contiguous()
    save_file(out, args.out)
    print(f"mtp: wrote {len(out)} tensors -> {args.out}")


if __name__ == "__main__":
    main()
