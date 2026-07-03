#!/usr/bin/env python3
"""
Convert Gemma-4 (E2B / E4B) checkpoints to ComfyUI text-encoder safetensors.

One source load serves every --job. The source may be a Hugging Face snapshot
directory (config.json + model.safetensors + tokenizer.json) or an already-mapped
ComfyUI bf16 safetensors; either way the bf16 ComfyUI layout is the base every job
builds on. Output keys are the original ComfyUI text-encoder keys (safetensors sorts
keys; no metadata). Quantized jobs emit native comfy_kitchen quantized-weight keys.

Precisions:
  bf16                          HF language_model/vision_tower/audio_tower/projectors
                                remapped to ComfyUI keys, every float tensor -> bfloat16,
                                KV-shared layer slots filled from the layer they share KV
                                with, tokenizer.json embedded as the uint8 tensor
                                `tokenizer_json`. This is the base for every other job.
  fp8_e4m3fn                    every tensor -> float8_e4m3fn
  int8                          language-model 2D .weight tensors (model.layers.* plus
                                model.embed_tokens and model.embed_tokens_per_layer) ->
                                stock-ComfyUI int8_tensorwise with ConvRot (per-channel
                                scale + weight-dtype group-Hadamard rotation, groupsize 256)
                                via comfy-quants _quantize_int8_tensorwise_per_row — the
                                canonical producer that byte-matches stock ComfyUI's own
                                save path. The vision tower, audio tower, and all projectors
                                stay bfloat16; weights whose in_features is not a multiple of
                                256 stay bfloat16.

Examples:
  # E4B: HF snapshot -> bf16 base and full-map int8 (one load serves both)
  python convert_gemma4_v2.py --src ~/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/<rev> \
      --job bf16:gemma4_e4b_it_bf16.safetensors \
      --job int8:gemma4_e4b_it_int8_convrot.safetensors

  # quantize straight off an existing ComfyUI bf16 file
  python convert_gemma4_v2.py --src gemma4_e4b_it_bf16.safetensors \
      --job int8:gemma4_e4b_it_int8_convrot.safetensors

A job may carry an expected SHA256 (PRECISION:OUT:SHA256) to verify the written file.

==========================================================================================
Provenance
==========================================================================================
Source checkpoints (Gemma Terms of Use), pinned to the exact HuggingFace revision:

  google/gemma-4-E4B-it   model.safetensors (bf16) + tokenizer.json + config.json
  google/gemma-4-E2B-it   model.safetensors (bf16) + tokenizer.json + config.json

int8 quantization is delegated to comfy-quants (comfy_quants.backends.
int8_tensorwise_model_export._quantize_int8_tensorwise_per_row) — the stock-ComfyUI
canonical producer — not comfy-kitchen from_float and not a hand-rolled quantizer. The
language-model / vision+audio-tower split matches the established ComfyUI Gemma-4
text-encoder int8 policy.

Outputs  ( sha256  file  <-  source, precision ):
  afe21e7c99d5a2ba52bc246a464d2458726204c3ce98ee81398204786ecab5ab  gemma4_e4b_it_bf16.safetensors  <- google/gemma-4-E4B-it, bf16  (byte-identical to the prior ComfyUI file)
  <int8 sha filled after a verified comfy-quants run>  gemma4_e4b_it_int8_convrot.safetensors  <- gemma4_e4b_it_bf16, int8_tensorwise convrot (comfy-quants canonical; 380 language-model layers incl. both embeddings; vision+audio towers bf16)
"""
import argparse
import collections
import hashlib
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Canonical quantization is comfy-quants (Comfy-Org's stock-ComfyUI-aligned offline
# quantizer). We call its int8_tensorwise producer directly — NOT comfy-kitchen
# from_float and NOT any hand-rolled path — so the int8 output byte-matches stock
# ComfyUI's own save path (ops.py:_quantized_weight_state_dict).
from comfy_quants.backends.int8_tensorwise_model_export import _quantize_int8_tensorwise_per_row
from comfy_quants.formats.int8_tensorwise import int8_tensorwise_checkpoint_quant_config

FP8 = torch.float8_e4m3fn
INT8_CONVROT_GROUPSIZE = 256  # power-of-4 Hadamard group; in_features must be a multiple

# HF -> ComfyUI text-encoder key prefixes.
PREFIX_MAP = [
    ("model.language_model.", "model."),
    ("model.audio_tower.", "audio_model."),
    ("model.vision_tower.", "vision_model."),
    ("model.embed_audio.", "audio_projector."),
    ("model.embed_vision.", "multi_modal_projector."),
]

# int8 policy: quantize the language model only. The vision tower, audio tower, and
# projectors stay bf16 ("keep the vision and audio towers in fp16, the rest goes int8").
INT8_KEEP_PREFIXES = (
    "vision_model.",
    "audio_model.",
    "audio_projector.",
    "multi_modal_projector.",
)
INT8_KEEP_EXACT = ("model.per_layer_model_projection",)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def map_key(key):
    for src, dst in PREFIX_MAP:
        if key.startswith(src):
            return dst + key[len(src):]
    raise ValueError(f"unmapped HF key: {key}")


def kv_share_sources(cfg):
    tc = cfg.get("text_config", cfg)
    n = tc["num_hidden_layers"]
    shared = tc["num_kv_shared_layers"]
    types = tc["layer_types"]
    first_shared = n - shared
    last_of_type = {}
    for i in range(first_shared):
        last_of_type[types[i]] = i
    return {i: last_of_type[types[i]] for i in range(first_shared, n)}


def load_base(src):
    """Return the ComfyUI bf16 base state dict, from an HF snapshot dir or a .safetensors."""
    if os.path.isdir(src):
        cfg = json.load(open(os.path.join(src, "config.json")))
        out = {}
        with safe_open(os.path.join(src, "model.safetensors"), framework="pt", device="cpu") as st:
            for key in st.keys():
                t = st.get_tensor(key)
                if t.dtype != torch.bfloat16 and t.is_floating_point():
                    t = t.to(torch.bfloat16)
                out[map_key(key)] = t
        filled = 0
        for shared_idx, src_idx in kv_share_sources(cfg).items():
            for leaf in ("self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_norm.weight"):
                dst_key = f"model.layers.{shared_idx}.{leaf}"
                if dst_key not in out:
                    out[dst_key] = out[f"model.layers.{src_idx}.{leaf}"].clone()
                    filled += 1
        tok = open(os.path.join(src, "tokenizer.json"), "rb").read()
        out["tokenizer_json"] = torch.tensor(list(tok), dtype=torch.uint8)
        print(f"mapped HF snapshot: {len(out)} tensors, filled {filled} kv-shared slots, "
              f"tokenizer {len(tok)} bytes")
        return out
    out = {}
    with safe_open(src, framework="pt", device="cpu") as st:
        for key in st.keys():
            out[key] = st.get_tensor(key)
    print(f"loaded ComfyUI bf16 base: {len(out)} tensors")
    return out


def marker_tensor(conf):
    # Encode the comfy_quant marker exactly like stock ComfyUI's save path: default
    # json.dumps separators + insertion order (comfy-quants' _marker_tensor).
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def int8_convrot_eligible(k, v):
    if not k.endswith(".weight") or v.dim() != 2:
        return False
    if v.shape[1] % INT8_CONVROT_GROUPSIZE != 0:
        return False
    base = k[:-len(".weight")]
    if base in INT8_KEEP_EXACT:
        return False
    return not k.startswith(INT8_KEEP_PREFIXES)


def quantize_int8_weight(k, v):
    base = k[:-len(".weight")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qweight, scale, rotated = _quantize_int8_tensorwise_per_row(
        v.contiguous().to(dev), convrot=True, group_size=INT8_CONVROT_GROUPSIZE)
    marker = int8_tensorwise_checkpoint_quant_config(
        convrot=rotated, convrot_groupsize=INT8_CONVROT_GROUPSIZE)
    return {
        f"{base}.weight": qweight.detach().to("cpu").contiguous(),
        f"{base}.weight_scale": scale.detach().to("cpu").contiguous(),
        f"{base}.comfy_quant": marker_tensor(marker),
    }


def cast(sd, precision):
    out = {}
    int8_quantized = int8_kept_policy = int8_kept_shape = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if precision == "bf16":
            out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        elif precision == "fp8_e4m3fn":
            out[k] = v.to(FP8) if v.is_floating_point() else v
        elif precision == "int8":
            if int8_convrot_eligible(k, v):
                out.update(quantize_int8_weight(k, v))
                int8_quantized += 1
            else:
                out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
                if k.endswith(".weight") and v.dim() == 2:
                    if k.startswith(INT8_KEEP_PREFIXES) or k[:-len(".weight")] in INT8_KEEP_EXACT:
                        int8_kept_policy += 1
                    elif v.shape[1] % INT8_CONVROT_GROUPSIZE != 0:
                        int8_kept_shape += 1
        else:
            raise SystemExit(f"unknown precision: {precision}")
    if precision == "int8":
        print(f"int8 quantized_weights={int8_quantized} kept_policy={int8_kept_policy} "
              f"kept_shape={int8_kept_shape} convrot_groupsize={INT8_CONVROT_GROUPSIZE}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert Gemma-4 to ComfyUI text-encoder safetensors.")
    ap.add_argument("--src", required=True, help="HF snapshot dir or a ComfyUI bf16 safetensors")
    ap.add_argument("--job", action="append", required=True, metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; one source load serves every job")
    ap.add_argument("--dump", action="store_true", help="print base tensor count and dtypes")
    args = ap.parse_args()

    base = load_base(args.src)

    if args.dump:
        dtypes = collections.Counter(str(v.dtype) for v in base.values() if torch.is_tensor(v))
        print(f"{len(base)} tensors, dtypes={dict(dtypes)}")

    mismatched = []
    for job in args.job:
        precision, sep, remainder = job.partition(":")
        if not sep or not remainder:
            raise SystemExit(f"invalid --job {job!r}; expected PRECISION:OUT[:SHA256]")
        out, expected = remainder, None
        head, sha_sep, tail = remainder.rpartition(":")
        if sha_sep and len(tail) == 64 and all(c in "0123456789abcdefABCDEF" for c in tail):
            out, expected = head, tail.lower()
        tensors = cast(base, precision)
        save_file(tensors, out)
        digest = sha256(out)
        verdict = "" if expected is None else ("  OK" if digest == expected else "  MISMATCH")
        print(f"{precision:14s} {digest}  {out}{verdict}")
        if expected is not None and digest != expected:
            mismatched.append(out)

    if mismatched:
        raise SystemExit(f"SHA256 mismatch: {', '.join(mismatched)}")


if __name__ == "__main__":
    main()
