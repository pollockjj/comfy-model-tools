#!/usr/bin/env python3
"""
Convert Gemma-4 (E2B / E4B) checkpoints to ComfyUI text-encoder safetensors.

V2: Kijai's V1 conversions, reskinned to the SeedVR2 --src/--job look. One source load
serves every --job; the source may be a Hugging Face snapshot directory or an existing
ComfyUI bf16 safetensors. Output is BYTE-IDENTICAL to the V1 tools (verified by SHA256);
the only change from V1 is the CLI/structure. Best-practice methods live in
convert_gemma4_v3.py.

Precisions:
  bf16                          HF language/vision/audio/projector layout remapped to
                                ComfyUI keys, kv-shared slots filled, tokenizer.json
                                embedded. Every float tensor -> bfloat16.
  fp8_scaled                    Kijai Gemma-4 FP8 V1: language-model 2D .weight tensors
                                with max(shape) >= 4096 (excluding norms) -> float8_e4m3fn,
                                per-tensor scale = amax / 416. Everything else bf16.
  int8_convrot                  Kijai int8 ConvRot V1 (quant_int8_convrot.quantize_convrot,
                                absmax, per-layer power-of-4 group Hadamard): the language
                                model (model.layers.* + embed_tokens + embed_tokens_per_layer)
                                -> int8_tensorwise; vision tower, audio tower, and projectors
                                stay bf16; weights whose in_features is not a multiple of the
                                group size stay bf16.

Examples:
  python convert_gemma4_v2.py --src <hf_dir_or_comfy_bf16> \
      --job bf16:gemma4_e4b_it_bf16.safetensors:afe21e7c99d5a2ba52bc246a464d2458726204c3ce98ee81398204786ecab5ab \
      --job fp8_scaled:gemma4_e4b_it_fp8_scaled.safetensors:bf0b4fa2e41a25684dc9e9b256cd505564f02fed09be3da95ce024e653e2c52b \
      --job int8_convrot:gemma4_e4b_it_int8_convrot.safetensors:057cbe0afd7fd30a56e7dddf526a0737558f278bb48e108e1cbd76b99571818b

A job may carry an expected SHA256 (PRECISION:OUT:SHA256) to verify the written file.

Reproducibility: bf16 and fp8_scaled are matmul-free (elementwise) and therefore
byte-identical on any machine/device (bf16 afe21e7c, fp8_scaled bf0b4fa2 match the prior
HF files). int8_convrot routes through the convrot rotation matmul, whose float
accumulation is compute-environment-specific: 4 environments produced 4 distinct shas
(avenger-5090 921326711, interceptor-3090 5fca0726, interceptor-CPU 057cbe0a,
firestorm-CPU d1d8f9ba). Per CLAUDE.md the canonical conversion runs on interceptor CPU,
so the canonical int8 sha is 057cbe0a; the older HF 921326711 is a superseded 5090 artifact.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Kijai's V1 int8 ConvRot quantizer, imported verbatim (provenance).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_int8_convrot import quantize_convrot_chunked, cq_tensor, best_gs

FP8_DTYPE = torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8_DTYPE)
FP8_MAX = 416.0  # Kijai Gemma-4 FP8 V1 convention
FP8_CONF = {"format": "float8_e4m3fn", "full_precision_matrix_mult": False}
INT8_CONVROT_GROUPSIZE = 256

PREFIX_MAP = [
    ("model.language_model.", "model."),
    ("model.audio_tower.", "audio_model."),
    ("model.vision_tower.", "vision_model."),
    ("model.embed_audio.", "audio_projector."),
    ("model.embed_vision.", "multi_modal_projector."),
]

# int8 policy: quantize the language model only (vision/audio towers + projectors stay bf16).
INT8_KEEP_PREFIXES = ("vision_model.", "audio_model.", "audio_projector.", "multi_modal_projector.")
INT8_KEEP_EXACT = ("model.per_layer_model_projection",)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def marker_tensor(conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def map_key(key):
    for src, dst in PREFIX_MAP:
        if key.startswith(src):
            return dst + key[len(src):]
    raise ValueError(f"unmapped HF key: {key}")


def kv_share_sources(cfg):
    tc = cfg.get("text_config", cfg)
    n, shared, types = tc["num_hidden_layers"], tc["num_kv_shared_layers"], tc["layer_types"]
    first_shared = n - shared
    last_of_type = {}
    for i in range(first_shared):
        last_of_type[types[i]] = i
    return {i: last_of_type[types[i]] for i in range(first_shared, n)}


def load_base(src):
    if os.path.isdir(src):
        cfg = json.load(open(os.path.join(src, "config.json")))
        out = {}
        with safe_open(os.path.join(src, "model.safetensors"), framework="pt", device="cpu") as st:
            for key in st.keys():
                t = st.get_tensor(key)
                if t.dtype != torch.bfloat16 and t.is_floating_point():
                    t = t.to(torch.bfloat16)
                out[map_key(key)] = t
        for shared_idx, src_idx in kv_share_sources(cfg).items():
            for leaf in ("self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_norm.weight"):
                dst = f"model.layers.{shared_idx}.{leaf}"
                if dst not in out:
                    out[dst] = out[f"model.layers.{src_idx}.{leaf}"].clone()
        tok = open(os.path.join(src, "tokenizer.json"), "rb").read()
        out["tokenizer_json"] = torch.tensor(list(tok), dtype=torch.uint8)
        print(f"mapped HF snapshot: {len(out)} tensors, tokenizer {len(tok)} bytes")
        return out
    out = {}
    with safe_open(src, framework="pt", device="cpu") as st:
        for key in st.keys():
            out[key] = st.get_tensor(key)
    print(f"loaded ComfyUI bf16 base: {len(out)} tensors")
    return out


def int8_convrot_eligible(k, v):
    if not k.endswith(".weight") or v.dim() != 2:
        return False
    if v.shape[1] % INT8_CONVROT_GROUPSIZE != 0:
        return False
    if k[:-len(".weight")] in INT8_KEEP_EXACT:
        return False
    return not k.startswith(INT8_KEEP_PREFIXES)


def should_quantize_fp8_scaled(k, v):
    # Kijai Gemma-4 FP8 V1 selection: large language-model 2D linears.
    return (k.startswith("model.") and k.endswith(".weight") and v.dim() == 2
            and "norm" not in k and max(v.shape) >= 4096)


def quantize_int8_weight(k, v):
    base = k[:-len(".weight")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gs = best_gs(v.shape[1])
    qd, scale, _cos, _relerr = quantize_convrot_chunked(v, gs, mseclip=False, device=dev)
    return {
        f"{base}.weight": qd.cpu().contiguous(),
        f"{base}.weight_scale": scale.cpu().contiguous(),
        f"{base}.comfy_quant": cq_tensor(gs),
    }


def quantize_fp8_scaled_weight(k, v):
    base = k[:-len(".weight")]
    w = v.float()
    scale = torch.max(torch.abs(w)) / FP8_MAX
    q = (w / scale).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(FP8_DTYPE)
    return {
        f"{base}.weight": q.cpu().contiguous(),
        f"{base}.weight_scale": scale.cpu(),
        f"{base}.comfy_quant": marker_tensor(FP8_CONF),
    }


def cast(sd, precision):
    out = {}
    nq = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if precision == "bf16":
            out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        elif precision == "fp8_scaled":
            if should_quantize_fp8_scaled(k, v):
                out.update(quantize_fp8_scaled_weight(k, v)); nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        elif precision == "int8_convrot":
            if int8_convrot_eligible(k, v):
                out.update(quantize_int8_weight(k, v)); nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        else:
            raise SystemExit(f"unknown precision: {precision}")
    if precision != "bf16":
        print(f"{precision}: quantized {nq} weights")
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert Gemma-4 to ComfyUI safetensors (V2, Kijai conversions).")
    ap.add_argument("--src", required=True, help="HF snapshot dir or a ComfyUI bf16 safetensors")
    ap.add_argument("--job", action="append", required=True, metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; one source load serves every job")
    args = ap.parse_args()

    base = load_base(args.src)
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
