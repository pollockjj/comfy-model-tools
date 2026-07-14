#!/usr/bin/env python3
"""
Convert Qwen3-VL 4B / 8B checkpoints to ComfyUI text-encoder safetensors.

Rev1 is the release-SHA reproduction script for the shipped Comfy-Org Qwen3-VL
text encoders. It deliberately preserves the released conversion conventions:

Precisions:
  bf16                          release BF16 repack. 4B carries no safetensors
                                metadata; 8B carries {"format": "pt"}.
  fp8_scaled                    language-model 2D projection weights only
                                (36 layers x 7 linears) -> float8_e4m3fn,
                                scalar scale = amax / 416, marker
                                full_precision_matrix_mult=false.
  nvfp4                         8B only: lm_head and embed_tokens -> the same
                                fp8_scaled path; the 36 x 7 projection weights
                                -> ComfyUI/comfy-kitchen TensorCoreNVFP4Layout
                                from BF16 arithmetic. Everything else bf16.

Examples:
  python convert_qwen3vl.py --src qwen3vl_4b_bf16.safetensors \
      --job bf16:qwen3vl_4b_bf16.safetensors:36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34 \
      --job fp8_scaled:qwen3vl_4b_fp8_scaled.safetensors:54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094

  PYTHONPATH=/home/johnj/dev_master/ComfyUI python convert_qwen3vl.py \
      --src qwen3vl_8b_bf16.safetensors \
      --job nvfp4:qwen3vl_8b_nvfp4.safetensors:e462e9e0c3b9313ae17f82040d7c77beb92d7aef3e40692d7803228dab7c3b98

A job may carry an expected SHA256 (PRECISION:OUT:SHA256) to verify the written
file. Source may be an existing released/ComfyUI bf16 safetensors or an HF
snapshot directory containing model.safetensors or model-*-of-*.safetensors.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FP8_DTYPE = torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8_DTYPE)
FP8_MAX = 416.0
FP8_CONF = {"format": "float8_e4m3fn", "full_precision_matrix_mult": False}
NVFP4_CONF = {"format": "nvfp4"}
LM_PROJECTION_LEAVES = (
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def marker_tensor(conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def iter_safetensors(src):
    if os.path.isfile(src):
        return [src]
    single = os.path.join(src, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    shards = sorted(glob.glob(os.path.join(src, "model-*-of-*.safetensors")))
    if not shards:
        raise SystemExit(f"no safetensors source found in {src}")
    return shards


def load_base(src):
    files = iter_safetensors(src)
    out = {}
    source_metadata = None
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as st:
            if source_metadata is None:
                source_metadata = st.metadata()
            for key in st.keys():
                t = st.get_tensor(key)
                if t.is_floating_point():
                    t = t.to(torch.bfloat16)
                out[key] = t
    print(f"loaded Qwen3-VL source: {len(out)} tensors, {len(files)} file(s)")
    return out, source_metadata


def remap_language_key(k):
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    return k


def normalized_base_key(k):
    return remap_language_key(k)[:-len(".weight")]


def layer_index(base):
    parts = base.split(".")
    if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def is_lm_projection_weight(k):
    mapped = remap_language_key(k)
    if not mapped.startswith("model.layers.") or not mapped.endswith(".weight"):
        return False
    idx = layer_index(mapped[:-len(".weight")])
    if idx is None or not (0 <= idx < 36):
        return False
    return any(mapped.endswith(leaf) for leaf in LM_PROJECTION_LEAVES)


def model_variant(sd):
    if any(k.startswith("model.language_model.") for k in sd):
        return "qwen3vl_4b"
    hidden = sd.get("model.visual.merger.linear_fc2.weight")
    if torch.is_tensor(hidden) and hidden.shape[0] == 4096:
        return "qwen3vl_8b"
    if "lm_head.weight" in sd:
        return "qwen3vl_8b"
    return "qwen3vl_4b"


def bf16_metadata(variant, source_metadata):
    if source_metadata and source_metadata.get("format") == "pt":
        return {"format": "pt"}
    if variant == "qwen3vl_8b":
        return {"format": "pt"}
    return None


def quantized_metadata():
    return {"format": "pt"}


def quantize_fp8_scaled_weight(k, v, out_key=None):
    mapped = out_key or remap_language_key(k)
    base = mapped[:-len(".weight")]
    w = v.float()
    scale = torch.max(torch.abs(w)) / FP8_MAX
    q = (w / scale).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(FP8_DTYPE)
    return {
        f"{base}.weight": q.cpu().contiguous(),
        f"{base}.weight_scale": scale.cpu(),
        f"{base}.comfy_quant": marker_tensor(FP8_CONF),
    }


def import_comfy_nvfp4(comfyui_root):
    if comfyui_root and comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)
    try:
        from comfy.quant_ops import TensorCoreNVFP4Layout
    except Exception as exc:
        raise SystemExit(
            "nvfp4 Rev1 reproduction requires ComfyUI with comfy-kitchen available. "
            "Run from the ComfyUI venv or pass --comfyui-root."
        ) from exc
    return TensorCoreNVFP4Layout


def quantize_nvfp4_weight(k, v, layout):
    mapped = remap_language_key(k)
    base = mapped[:-len(".weight")]
    qweight, params = layout.quantize(v.contiguous())
    return {
        f"{base}.weight": qweight.detach().to("cpu").contiguous(),
        f"{base}.weight_scale": params.block_scale.detach().to("cpu").contiguous(),
        f"{base}.weight_scale_2": params.scale.detach().to("cpu").to(torch.float32).contiguous(),
        f"{base}.comfy_quant": marker_tensor(NVFP4_CONF),
    }


def cast(sd, precision, comfyui_root):
    out = {}
    nq = 0
    variant = model_variant(sd)
    nvfp4_layout = import_comfy_nvfp4(comfyui_root) if precision == "nvfp4" else None
    if precision == "nvfp4" and variant != "qwen3vl_8b":
        raise SystemExit("released nvfp4 artifact exists for Qwen3VL-8B only")

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        mapped = remap_language_key(k)
        if precision == "bf16":
            out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        elif precision == "fp8_scaled":
            if is_lm_projection_weight(k):
                out.update(quantize_fp8_scaled_weight(k, v)); nq += 1
            else:
                out[mapped] = v.to(torch.bfloat16) if v.is_floating_point() else v
        elif precision == "nvfp4":
            if mapped in ("lm_head.weight", "model.embed_tokens.weight"):
                out.update(quantize_fp8_scaled_weight(k, v, out_key=mapped)); nq += 1
            elif is_lm_projection_weight(k):
                out.update(quantize_nvfp4_weight(k, v, nvfp4_layout)); nq += 1
            else:
                out[mapped] = v.to(torch.bfloat16) if v.is_floating_point() else v
        else:
            raise SystemExit(f"unknown precision: {precision}")
    if precision in ("fp8_scaled", "nvfp4"):
        print(f"{precision}: quantized {nq} weights")
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert Qwen3-VL to ComfyUI safetensors (Rev1 release-SHA reproduction).")
    ap.add_argument("--src", required=True, help="HF snapshot dir or ComfyUI/released bf16 safetensors")
    ap.add_argument("--job", action="append", required=True, metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; one source load serves every job")
    ap.add_argument("--comfyui-root", default="/home/johnj/dev_master/ComfyUI",
                    help="ComfyUI checkout used for Rev1 nvfp4 reproduction")
    args = ap.parse_args()

    base, source_metadata = load_base(args.src)
    variant = model_variant(base)
    mismatched = []
    for job in args.job:
        precision, sep, remainder = job.partition(":")
        if not sep or not remainder:
            raise SystemExit(f"invalid --job {job!r}; expected PRECISION:OUT[:SHA256]")
        out, expected = remainder, None
        head, sha_sep, tail = remainder.rpartition(":")
        if sha_sep and len(tail) == 64 and all(c in "0123456789abcdefABCDEF" for c in tail):
            out, expected = head, tail.lower()
        tensors = cast(base, precision, args.comfyui_root)
        metadata = bf16_metadata(variant, source_metadata) if precision == "bf16" else quantized_metadata()
        save_file(tensors, out, metadata=metadata)
        digest = sha256(out)
        verdict = "" if expected is None else ("  OK" if digest == expected else "  MISMATCH")
        print(f"{precision:14s} {digest}  {out}{verdict}")
        if expected is not None and digest != expected:
            mismatched.append(out)

    if mismatched:
        raise SystemExit(f"SHA256 mismatch: {', '.join(mismatched)}")


if __name__ == "__main__":
    main()
