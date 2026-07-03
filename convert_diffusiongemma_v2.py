#!/usr/bin/env python3
"""
Convert DiffusionGemma-26B-A4B checkpoints to ComfyUI text-encoder safetensors.

One source load serves every --job. The source may be a Hugging Face snapshot
directory (11 model-*-of-*.safetensors shards + config.json + tokenizer.json) or an
already-mapped ComfyUI bf16 safetensors; either way the bf16 ComfyUI layout is the
base every job builds on. Output keys are the ComfyUI text-encoder keys (safetensors
sorts keys; no metadata). Quantized jobs emit native comfy_kitchen quantized-weight
keys.

Precisions:
  bf16                          HF shards remapped to ComfyUI keys (lm_head dropped,
                                expert banks suffixed .weight), every float tensor ->
                                bfloat16, tokenizer.json embedded as `tokenizer_json`.
                                This is the base for every other job.
  fp8_e4m3fn                    per-token 2D Linear weights (attention q/k/v/o_proj and
                                dense gate/up/down_proj) -> float8_e4m3fn via comfy_kitchen
                                QuantizedTensor.from_float; the 3D grouped expert banks ->
                                per-expert absmax float8_e4m3fn (one scale per expert).
                                Embeddings, router, self-conditioning, vision encoder,
                                norms/scalars, and the tokenizer stay bfloat16.

Examples:
  # HF snapshot -> bf16 base and fp8
  python convert_diffusiongemma_v2.py \
      --src ~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/<rev> \
      --job bf16:diffusiongemma_comfy_bf16.safetensors \
      --job fp8_e4m3fn:diffusiongemma_comfy_fp8.safetensors

  # quantize straight off an existing ComfyUI bf16 file
  python convert_diffusiongemma_v2.py --src diffusiongemma_comfy_bf16.safetensors \
      --job fp8_e4m3fn:diffusiongemma_comfy_fp8.safetensors

A job may carry an expected SHA256 (PRECISION:OUT:SHA256) to verify the written file.

==========================================================================================
Provenance
==========================================================================================
Source checkpoint (Gemma Terms of Use), pinned to the exact HuggingFace revision:

  google/diffusiongemma-26B-A4B-it @ 0f28bc42f588fbd8f71e08102b1c3960298a1358
    11 shards + config.json + tokenizer.json; 30 decoder layers, 128 experts.

fp8 uses comfy_kitchen QuantizedTensor.from_float for 2D Linear weights (the documented
canonical kernel) and per-expert absmax fp8 for the 3D grouped expert banks.

Outputs  ( sha256  file  <-  source, precision ):
  495d347e1b6c1aa13338741a17d1f5632f3ad4adb11f85f8eeb6ec026db418d1  diffusiongemma_comfy_bf16.safetensors  <- google/diffusiongemma-26B-A4B-it, bf16  (byte-identical to the prior ComfyUI file)
"""
import argparse
import collections
import hashlib
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_LAYOUT = "TensorCoreFP8Layout"
FP8_FORMAT = "float8_e4m3fn"
FP8_DTYPE = torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8_DTYPE)

EXPECTED_SHARDS = 11
EXPECTED_LAYERS = 30
EXPECTED_EXPERTS = 128
EXPERT_HF_SUFFIXES = (".experts.gate_up_proj", ".experts.down_proj")
EXPERT_COMFY_SUFFIXES = (".experts.gate_up_proj.weight", ".experts.down_proj.weight")
ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_hf_key(key):
    if key.startswith("lm_head."):
        return None
    if key.endswith(EXPERT_HF_SUFFIXES):
        return f"{key}.weight"
    return key


def load_base(src):
    """Return the ComfyUI bf16 base state dict, from an HF snapshot dir or a .safetensors."""
    if os.path.isdir(src):
        cfg = json.load(open(os.path.join(src, "config.json")))
        tc = cfg.get("text_config", {})
        if tc.get("num_hidden_layers") != EXPECTED_LAYERS or tc.get("num_experts") != EXPECTED_EXPERTS:
            raise SystemExit(f"unexpected DG config: layers={tc.get('num_hidden_layers')} "
                             f"experts={tc.get('num_experts')}")
        shards = sorted(__import__("glob").glob(os.path.join(src, "model-*-of-*.safetensors")))
        if len(shards) != EXPECTED_SHARDS:
            raise SystemExit(f"expected {EXPECTED_SHARDS} HF shards, found {len(shards)}")
        out = {}
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    out_key = normalize_hf_key(key)
                    if out_key is None:
                        continue
                    if out_key in out:
                        raise SystemExit(f"duplicate output key after HF mapping: {out_key}")
                    out[out_key] = f.get_tensor(key)
        tok = open(os.path.join(src, "tokenizer.json"), "rb").read()
        out["tokenizer_json"] = torch.tensor(list(tok), dtype=torch.uint8)
        print(f"mapped HF snapshot: {len(out)} tensors, {len(shards)} shards, tokenizer {len(tok)} bytes")
        return out
    out = {}
    with safe_open(src, framework="pt", device="cpu") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    print(f"loaded ComfyUI bf16 base: {len(out)} tensors")
    return out


def comfy_quant_tensor(conf):
    return torch.tensor(list(json.dumps(conf, sort_keys=True).encode("utf-8")), dtype=torch.uint8)


def is_expert_bank(key):
    return key.endswith(EXPERT_COMFY_SUFFIXES)


def is_attn_or_mlp_2d(key):
    if not key.startswith("model.decoder.layers.") or not key.endswith(".weight"):
        return False
    return (any(f".self_attn.{n}.weight" in key for n in ATTN_PROJECTIONS)
            or any(f".mlp.{n}.weight" in key for n in MLP_PROJECTIONS))


def quantize_2d_fp8(key, tensor):
    try:
        from comfy_kitchen.tensor import QuantizedTensor
    except ImportError as e:
        raise SystemExit("fp8_e4m3fn requires comfy-kitchen") from e
    qt = QuantizedTensor.from_float(tensor.contiguous(), FP8_LAYOUT)
    tensors = qt.state_dict(key)
    tensors[key.replace(".weight", ".comfy_quant")] = comfy_quant_tensor({"format": FP8_FORMAT})
    return tensors


def quantize_expert_bank(key, tensor):
    if tensor.dim() != 3 or tensor.shape[0] != EXPECTED_EXPERTS:
        raise SystemExit(f"expert bank {key} not [E,*,*] with E={EXPECTED_EXPERTS}: {tuple(tensor.shape)}")
    qdata = torch.empty(tensor.shape, dtype=FP8_DTYPE)
    scales = torch.empty((tensor.shape[0],), dtype=torch.float32)
    for e in range(tensor.shape[0]):
        expert = tensor[e].float()
        scale = torch.amax(expert.abs()) / FP8_INFO.max
        if scale.item() == 0:
            raise SystemExit(f"zero fp8 scale for {key} expert {e}")
        scales[e] = scale
        qdata[e] = (expert / scale).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(FP8_DTYPE)
    return {
        key: qdata,
        key.replace(".weight", ".weight_scale"): scales,
        key.replace(".weight", ".comfy_quant"): comfy_quant_tensor(
            {"format": FP8_FORMAT, "num_experts": tensor.shape[0], "scale_granularity": "per_expert"}),
    }


def cast(sd, precision):
    out = {}
    q2d = qexp = kept = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if precision == "bf16":
            out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        elif precision == "fp8_e4m3fn":
            if is_expert_bank(k):
                out.update(quantize_expert_bank(k, v)); qexp += 1
            elif is_attn_or_mlp_2d(k):
                out.update(quantize_2d_fp8(k, v)); q2d += 1
            else:
                out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
                kept += 1
        else:
            raise SystemExit(f"unknown precision: {precision}")
    if precision == "fp8_e4m3fn":
        if qexp != EXPECTED_LAYERS * 2:
            raise SystemExit(f"expert coverage {qexp} != {EXPECTED_LAYERS * 2}")
        print(f"fp8_e4m3fn expert_banks={qexp} linear2d={q2d} kept_bf16={kept}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert DiffusionGemma to ComfyUI text-encoder safetensors.")
    ap.add_argument("--src", required=True, help="HF snapshot dir or a ComfyUI bf16 safetensors")
    ap.add_argument("--job", action="append", required=True, metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; one source load serves every job")
    ap.add_argument("--dump", action="store_true", help="print base tensor count and dtypes")
    args = ap.parse_args()

    base = load_base(args.src)
    if args.dump:
        dtypes = collections.Counter(str(v.dtype) for v in base.values() if torch.is_tensor(v))
        experts = sum(1 for k in base if is_expert_bank(k))
        print(f"{len(base)} tensors, dtypes={dict(dtypes)}, expert_banks={experts}")

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
