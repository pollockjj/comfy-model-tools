#!/usr/bin/env python3
"""
Convert DiffusionGemma-26B-A4B checkpoints to ComfyUI text-encoder safetensors.

V3 = V2 + int8 convrot. DiffusionGemma's int8 convrot job is NOT built yet — its 3D MoE
expert banks have no comfy-quants routine (a separate build) — so V3 currently carries
only the bf16 and fp8 jobs and is byte-identical to V2 (495d347e / 3d26c504). The bf16 and
fp8 code below is V2's exact code; when the int8 job lands it is added alongside, leaving
bf16/fp8 untouched.

Keys are kept in HF naming (model.decoder.*, model.encoder.*); the only structural change
is renaming the fused expert banks to <bank>.weight (comfy.ops.MoEExperts) and embedding
tokenizer.json. lm_head.* is dropped.

Precisions:
  bf16                          every float tensor -> bfloat16; expert banks -> <bank>.weight.
  fp8                           Kijai FP8 V1 (max_value=416): expert banks (3D) -> float8_e4m3fn
                                with per-expert scale (amax over dims 1,2 / 416); large 2D
                                text-backbone weights (model.decoder.layers.*, max(shape) >= 4096,
                                non-norm) -> float8_e4m3fn per-tensor scale (amax / 416).
                                Everything else -> bfloat16.

Examples:
  python convert_diffusiongemma_v2.py \
      --src ~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/<rev> \
      --job bf16:diffusiongemma_comfy_bf16.safetensors:495d347e1b6c1aa13338741a17d1f5632f3ad4adb11f85f8eeb6ec026db418d1 \
      --job fp8:diffusiongemma_comfy_fp8.safetensors:3d26c504c323bc78fa2d51dbc8433ba4ccf45dcb015b46122d2e37e4c4496015

A job may carry an expected SHA256 (PRECISION:OUT:SHA256) to verify the written file.

Reproducibility: both jobs are matmul-free (bf16 passthrough; fp8 is per-expert / per-tensor
amax/416, elementwise), so output is byte-identical on any machine/device. Verified on
interceptor CPU: bf16 495d347e and fp8 3d26c504 both match the shipped HF files exactly.
"""
import argparse
import glob
import hashlib
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_DTYPE = torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8_DTYPE)
FP8_MAX = 416.0  # Kijai DiffusionGemma FP8 V1 convention
EXPERT_HF_SUFFIXES = (".experts.gate_up_proj", ".experts.down_proj")
EXPERT_WEIGHT_SUFFIXES = (".experts.gate_up_proj.weight", ".experts.down_proj.weight")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def marker_tensor(conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def load_base(src):
    if not os.path.isdir(src):
        out = {}
        with safe_open(src, framework="pt", device="cpu") as f:
            for key in f.keys():
                out[key] = f.get_tensor(key)
        print(f"loaded ComfyUI bf16 base: {len(out)} tensors")
        return out
    shards = sorted(glob.glob(os.path.join(src, "model-*-of-*.safetensors")))
    if not shards:
        raise SystemExit(f"no shards found in {src}")
    out = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("lm_head."):
                    continue
                out_key = f"{k}.weight" if k.endswith(EXPERT_HF_SUFFIXES) else k
                out[out_key] = f.get_tensor(k)
    tok = open(os.path.join(src, "tokenizer.json"), "rb").read()
    out["tokenizer_json"] = torch.tensor(list(tok), dtype=torch.uint8)
    print(f"mapped HF snapshot: {len(out)} tensors, {len(shards)} shards, tokenizer {len(tok)} bytes")
    return out


def is_expert_bank(k):
    return k.endswith(EXPERT_WEIGHT_SUFFIXES)


def should_quantize_2d(k, v):
    return (k.startswith("model.decoder.layers.") and k.endswith(".weight") and v.dim() == 2
            and "norm" not in k and max(v.shape) >= 4096)


def quantize_bank(k, w):
    base = k[:-len(".weight")]
    w = w.float()
    scale = torch.amax(torch.abs(w), dim=(1, 2)) / FP8_MAX
    w_q = (w / scale[:, None, None]).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(FP8_DTYPE)
    return {
        f"{base}.weight": w_q,
        f"{base}.weight_scale": scale,
        f"{base}.comfy_quant": marker_tensor({"format": "float8_e4m3fn", "num_experts": w.shape[0]}),
    }


def quantize_2d(k, w):
    base = k[:-len(".weight")]
    w = w.float()
    scale = torch.max(torch.abs(w)) / FP8_MAX
    w_q = (w / scale).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(FP8_DTYPE)
    return {
        f"{base}.weight": w_q,
        f"{base}.weight_scale": scale,
        f"{base}.comfy_quant": marker_tensor({"format": "float8_e4m3fn"}),
    }


def cast(sd, precision):
    out = {}
    nq = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if precision == "bf16":
            out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        elif precision == "fp8":
            if is_expert_bank(k):
                out.update(quantize_bank(k, v)); nq += 1
            elif should_quantize_2d(k, v):
                out.update(quantize_2d(k, v)); nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        else:
            raise SystemExit(f"unknown precision: {precision}")
    if precision == "fp8":
        print(f"fp8: quantized {nq} weights")
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert DiffusionGemma to ComfyUI safetensors (V2, Kijai conversion).")
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
