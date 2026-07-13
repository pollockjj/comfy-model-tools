#!/usr/bin/env python3
"""
Convert Gemma-4 (E2B / E4B) checkpoints to ComfyUI text-encoder safetensors.

V3 = V2 + int8 convrot. The bf16 and fp8_scaled jobs ARE V2's exact code, byte-identical
to V2 (afe21e7c / bf0b4fa2 — the shipped Comfy-Org / prior ComfyUI files). The only
addition over V2 is the int8 job. One source load serves every --job; the source may be a
Hugging Face snapshot directory or an existing ComfyUI bf16 safetensors.

Precisions:
  bf16                          HF language/vision/audio/projector layout remapped to
                                ComfyUI keys, kv-shared slots filled, tokenizer.json
                                embedded. Every float tensor -> bfloat16. (identical to V2)
  fp8_scaled                    Kijai Gemma-4 FP8 V1: language-model 2D .weight tensors
                                with max(shape) >= 4096 (excluding norms) -> float8_e4m3fn,
                                per-tensor scale = amax / 416. Everything else bf16. (identical to V2)
  int8                          language-model 2D .weight tensors (in_features multiple of
                                256; vision/audio towers + projectors excluded) -> stock-
                                ComfyUI int8_tensorwise convrot (per-row scale + group-
                                Hadamard rotation, groupsize 256) via comfy-quants
                                _quantize_int8_tensorwise_per_row. int8 convrot is a
                                torch.matmul, byte-reproducible only within one fixed
                                environment; the canonical surface is interceptor CPU
                                (--device cpu, CUDA_VISIBLE_DEVICES=).

Examples:
  python convert_gemma4.py --src <hf_dir_or_comfy_bf16> \
      --job bf16:gemma4_e4b_it_bf16.safetensors:afe21e7c99d5a2ba52bc246a464d2458726204c3ce98ee81398204786ecab5ab \
      --job fp8_scaled:gemma4_e4b_it_fp8_scaled.safetensors:bf0b4fa2e41a25684dc9e9b256cd505564f02fed09be3da95ce024e653e2c52b

  CUDA_VISIBLE_DEVICES= python convert_gemma4.py --device cpu --src gemma4_e4b_it_bf16.safetensors \
      --job int8:gemma4_e4b_it_int8_convrot.safetensors:065ea4422aa107c7133e9cf530582d6ba65057d089e35824e1d06da20960818c

A job may carry an expected SHA256 (PRECISION:OUT:SHA256) to verify the written file. bf16
and fp8_scaled are matmul-free (byte-identical on any machine); int8 convrot is interceptor-
CPU canonical.
"""
import argparse
import hashlib
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# int8 convrot is delegated to comfy-quants' stock-ComfyUI producer (byte-matches
# ComfyUI's own save path). bf16 and fp8_scaled below are V2's exact code, untouched.
from comfy_quants.backends.int8_tensorwise_model_export import _quantize_int8_tensorwise_per_row
from comfy_quants.formats.int8_tensorwise import int8_tensorwise_checkpoint_quant_config

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

# int8 policy: language model only; vision/audio towers + projectors stay bf16.
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


def should_quantize_fp8_scaled(k, v):
    # Kijai Gemma-4 FP8 V1 selection: large language-model 2D linears.
    return (k.startswith("model.") and k.endswith(".weight") and v.dim() == 2
            and "norm" not in k and max(v.shape) >= 4096)


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


def int8_convrot_eligible(k, v):
    if not k.endswith(".weight") or v.dim() != 2:
        return False
    if v.shape[1] % INT8_CONVROT_GROUPSIZE != 0:
        return False
    base = k[:-len(".weight")]
    if base in INT8_KEEP_EXACT:
        return False
    return not k.startswith(INT8_KEEP_PREFIXES)


def quantize_int8_weight(k, v, dev):
    base = k[:-len(".weight")]
    qweight, scale, rotated = _quantize_int8_tensorwise_per_row(
        v.contiguous().to(dev), convrot=True, group_size=INT8_CONVROT_GROUPSIZE)
    marker = int8_tensorwise_checkpoint_quant_config(
        convrot=rotated, convrot_groupsize=INT8_CONVROT_GROUPSIZE)
    return {
        f"{base}.weight": qweight.detach().to("cpu").contiguous(),
        f"{base}.weight_scale": scale.detach().to("cpu").contiguous(),
        f"{base}.comfy_quant": marker_tensor(marker),
    }


def cast(sd, precision, device="cpu"):
    out = {}
    nq = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if precision == "bf16":
            out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        elif precision == "fp8_scaled":
            if should_quantize_fp8_scaled(k, v):
                out.update(quantize_fp8_scaled_weight(k, v))
                nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        elif precision == "int8":
            if int8_convrot_eligible(k, v):
                out.update(quantize_int8_weight(k, v, device))
                nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        else:
            raise SystemExit(f"unknown precision: {precision}")
    if precision == "fp8_scaled":
        print(f"fp8_scaled: quantized {nq} weights")
    if precision == "int8":
        print(f"int8: quantized {nq} weights (convrot groupsize {INT8_CONVROT_GROUPSIZE})")
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert Gemma-4 to ComfyUI safetensors (V3 = V2 + int8 convrot).")
    ap.add_argument("--src", required=True, help="HF snapshot dir or a ComfyUI bf16 safetensors")
    ap.add_argument("--job", action="append", required=True, metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; one source load serves every job")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="int8 convrot device; cpu is the canonical byte-reproducible target "
                         "(interceptor-CPU mandate), cuda is throwaway-speed only")
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
        tensors = cast(base, precision, args.device)
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
