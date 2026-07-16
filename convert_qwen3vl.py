#!/usr/bin/env python3
"""
Convert Qwen3-VL 4B / 8B checkpoints to ComfyUI text-encoder safetensors.

Rev0 deterministic release converter for the shipped Comfy-Org Qwen3-VL text
encoders.

Rev0 accepts only the pinned original Qwen checkpoints listed below and always
checks the complete output file against the embedded canonical release SHA256.

Pinned sources:
  Qwen/Qwen3-VL-4B-Instruct @ ebb281ec70b05090aa6165b016eac8ec08e71b17
  Qwen/Qwen3-VL-8B-Instruct @ 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

Released conversion conventions:

Precisions:
  bf16                          release BF16 repack. 4B carries no safetensors
                                metadata; 8B carries {"format": "pt"}.
  fp8_scaled                    language-model 2D projection weights only
                                (36 layers x 7 linears) -> float8_e4m3fn,
                                scalar scale = amax * float32(1 / 416), marker
                                full_precision_matrix_mult=false.
  nvfp4                         8B only: lm_head and embed_tokens -> the same
                                fp8_scaled path; the 36 x 7 projection weights
                                -> released NVFP4 packing from BF16 arithmetic,
                                including CUDA reciprocal-multiply ordering.
                                Everything else bf16.

Examples:
  python convert_qwen3vl.py --src Qwen3-VL-4B-Instruct-snapshot \
      --job bf16:qwen3vl_4b_bf16.safetensors \
      --job fp8_scaled:qwen3vl_4b_fp8_scaled.safetensors

  PYTHONPATH=/path/to/ComfyUI python convert_qwen3vl.py \
      --src Qwen3-VL-8B-Instruct-snapshot \
      --job nvfp4:qwen3vl_8b_nvfp4.safetensors \
      --comfyui-root /path/to/ComfyUI

The optional SHA256 in PRECISION:OUT:SHA256 is accepted only when it equals the
embedded canonical SHA. Source must be the pinned original HF snapshot.
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
FP8_RECIPROCAL = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
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

PINNED_SOURCE_SHAS = {
    "qwen3vl_4b": (
        "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
        "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
    ),
    "qwen3vl_8b": (
        "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
        "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
        "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
        "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
    ),
}

EXPECTED_SHAS = {
    ("qwen3vl_4b", "bf16"): "36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34",
    ("qwen3vl_4b", "fp8_scaled"): "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094",
    ("qwen3vl_8b", "bf16"): "68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9",
    ("qwen3vl_8b", "fp8_scaled"): "4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd",
    ("qwen3vl_8b", "nvfp4"): "e462e9e0c3b9313ae17f82040d7c77beb92d7aef3e40692d7803228dab7c3b98",
}

NVFP4_RCP_ULP_UP_MANTISSAS = (
    0x510000, 0x5B0000, 0x6D0000, 0x730000, 0x760000, 0x790000,
    0xD10000, 0xDB0000, 0xED0000, 0xF30000, 0xF60000, 0xF90000,
)
NVFP4_RCP_ULP_DOWN_MANTISSAS = (0x160000, 0x960000)


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


def identify_pinned_source(files):
    digests = tuple(sha256(file) for file in files)
    for variant, expected in PINNED_SOURCE_SHAS.items():
        if digests == expected:
            print(f"verified pinned source: {variant}")
            return variant
    raise SystemExit(f"unrecognized source shard SHA256 tuple: {digests}")


def load_base(files):
    out = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as st:
            for key in st.keys():
                t = st.get_tensor(key)
                if t.is_floating_point():
                    t = t.to(torch.bfloat16)
                out[key] = t
    print(f"loaded Qwen3-VL source: {len(out)} tensors, {len(files)} file(s)")
    return out


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
    hidden = sd.get("model.visual.merger.linear_fc2.weight")
    if torch.is_tensor(hidden) and hidden.shape[0] == 4096:
        return "qwen3vl_8b"
    if torch.is_tensor(hidden) and hidden.shape[0] == 2560:
        return "qwen3vl_4b"
    raise SystemExit("source tensors do not identify a supported Qwen3-VL variant")


def bf16_metadata(variant):
    return {"format": "pt"} if variant == "qwen3vl_8b" else None


def quantized_metadata():
    return {"format": "pt"}


def quantize_fp8_scaled_weight(k, v, out_key=None):
    mapped = out_key or remap_language_key(k)
    base = mapped[:-len(".weight")]
    w = v.float()
    scale = torch.max(torch.abs(w)) * FP8_RECIPROCAL
    q = (w / scale).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(FP8_DTYPE)
    return {
        f"{base}.weight": q.cpu().contiguous(),
        f"{base}.weight_scale": scale.cpu(),
        f"{base}.comfy_quant": marker_tensor(FP8_CONF),
    }


def emulate_released_nvfp4_fast_reciprocal(scale):
    reciprocal = torch.reciprocal(scale)
    up = torch.nextafter(reciprocal, torch.full_like(reciprocal, float("inf")))
    down = torch.nextafter(reciprocal, torch.full_like(reciprocal, float("-inf")))
    mantissa = scale.view(torch.int32) & 0x7FFFFF
    up_mantissas = torch.tensor(NVFP4_RCP_ULP_UP_MANTISSAS, dtype=torch.int32, device=scale.device)
    down_mantissas = torch.tensor(NVFP4_RCP_ULP_DOWN_MANTISSAS, dtype=torch.int32, device=scale.device)
    use_up = (mantissa.unsqueeze(-1) == up_mantissas).any(dim=-1)
    use_down = (mantissa.unsqueeze(-1) == down_mantissas).any(dim=-1)
    return torch.where(use_up, up, torch.where(use_down, down, reciprocal))


def import_nvfp4_block_layout(comfyui_root):
    if comfyui_root and comfyui_root not in sys.path:
        sys.path.insert(0, comfyui_root)
    try:
        from comfy_kitchen.float_utils import _f32_to_floatx_unpacked, pack_uint4, to_blocked
    except Exception as exc:
        raise SystemExit(
            "nvfp4 Rev0 reproduction requires comfy-kitchen's block layout. "
            "Run from the ComfyUI venv or pass --comfyui-root."
        ) from exc
    return to_blocked, _f32_to_floatx_unpacked, pack_uint4


def quantize_nvfp4_weight(k, v, nvfp4_helpers):
    mapped = remap_language_key(k)
    base = mapped[:-len(".weight")]
    to_blocked, f32_to_floatx, pack_uint4 = nvfp4_helpers
    rows, cols = v.shape
    if rows % 16 or cols % 16:
        raise SystemExit(f"released NVFP4 tensor is not 16x16 aligned: {mapped} {tuple(v.shape)}")

    # comfy-kitchen v0.2.9's released CUDA kernel multiplies by the reciprocal
    # encode scale. Its eager CPU fallback divides by the decode scale. Those
    # operations diverge at FP4 boundaries, so preserve the CUDA operation order.
    scale = (torch.amax(v.abs()) / (448.0 * 6.0)).to(torch.float32)
    blocked = v.reshape(rows, -1, 16)
    block_scale = torch.amax(blocked.abs(), dim=-1).to(torch.float32) / 6.0
    scaled_block_scale = torch.clamp(block_scale / scale, max=448.0)
    block_scale_fp8 = scaled_block_scale.to(torch.float8_e4m3fn)
    total_scale = scale * block_scale_fp8.to(torch.float32)
    zero_scale = total_scale == 0
    safe_scale = torch.where(zero_scale, torch.ones_like(total_scale), total_scale)
    encode_scale = emulate_released_nvfp4_fast_reciprocal(safe_scale)
    normalized = blocked.to(torch.float32) * encode_scale.unsqueeze(-1)
    normalized = torch.where(zero_scale.unsqueeze(-1), torch.zeros_like(normalized), normalized)
    normalized = normalized.clamp(-6.0, 6.0).view(rows, cols)

    unpacked = f32_to_floatx(normalized, 2, 1)
    qweight = pack_uint4(unpacked, hi_first=True).contiguous()
    stored_block_scale = to_blocked(block_scale_fp8, flatten=False)
    return {
        f"{base}.weight": qweight.cpu(),
        f"{base}.weight_scale": stored_block_scale.cpu().contiguous(),
        f"{base}.weight_scale_2": scale.cpu().contiguous(),
        f"{base}.comfy_quant": marker_tensor(NVFP4_CONF),
    }


def cast(sd, precision, comfyui_root):
    out = {}
    nq = 0
    variant = model_variant(sd)
    nvfp4_helpers = import_nvfp4_block_layout(comfyui_root) if precision == "nvfp4" else None
    if precision == "nvfp4" and variant != "qwen3vl_8b":
        raise SystemExit("released nvfp4 artifact exists for Qwen3VL-8B only")

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        mapped = remap_language_key(k)
        if precision == "bf16":
            out_key = mapped if variant == "qwen3vl_8b" else k
            out[out_key] = v.to(torch.bfloat16) if v.is_floating_point() else v
        elif precision == "fp8_scaled":
            if is_lm_projection_weight(k):
                out.update(quantize_fp8_scaled_weight(k, v)); nq += 1
            else:
                out[mapped] = v.to(torch.bfloat16) if v.is_floating_point() else v
        elif precision == "nvfp4":
            if mapped in ("lm_head.weight", "model.embed_tokens.weight"):
                tensors = quantize_fp8_scaled_weight(k, v, out_key=mapped)
                out.update(tensors); nq += 1
            elif is_lm_projection_weight(k):
                out.update(quantize_nvfp4_weight(k, v, nvfp4_helpers)); nq += 1
            else:
                out[mapped] = v.to(torch.bfloat16) if v.is_floating_point() else v
        else:
            raise SystemExit(f"unknown precision: {precision}")
    if precision in ("fp8_scaled", "nvfp4"):
        print(f"{precision}: quantized {nq} weights")
    return out


def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL Rev0 deterministic release converter.")
    ap.add_argument("--src", required=True, help="pinned original HF snapshot directory")
    ap.add_argument("--job", action="append", required=True, metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; one source load serves every job")
    ap.add_argument("--comfyui-root",
                    help="optional ComfyUI checkout that makes comfy-kitchen importable for nvfp4")
    args = ap.parse_args()

    files = iter_safetensors(args.src)
    variant = identify_pinned_source(files)
    base = load_base(files)
    detected_variant = model_variant(base)
    if detected_variant != variant:
        raise SystemExit(f"source identity mismatch: hashes={variant}, tensors={detected_variant}")
    mismatched = []
    for job in args.job:
        precision, sep, remainder = job.partition(":")
        if not sep or not remainder:
            raise SystemExit(f"invalid --job {job!r}; expected PRECISION:OUT[:SHA256]")
        out, expected = remainder, None
        head, sha_sep, tail = remainder.rpartition(":")
        if sha_sep and len(tail) == 64 and all(c in "0123456789abcdefABCDEF" for c in tail):
            out, expected = head, tail.lower()
        canonical = EXPECTED_SHAS.get((variant, precision))
        if canonical is None:
            raise SystemExit(f"no released artifact for {variant} {precision}")
        if expected is not None and expected != canonical:
            raise SystemExit(f"caller SHA256 does not equal canonical SHA256 for {variant} {precision}")
        expected = canonical
        tensors = cast(base, precision, args.comfyui_root)
        metadata = bf16_metadata(variant) if precision == "bf16" else quantized_metadata()
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
