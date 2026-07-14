#!/usr/bin/env python3
"""
Convert DiffusionGemma-26B-A4B checkpoints to ComfyUI text-encoder safetensors.

V3 = V2 + int8 convrot + fused MXFP8 experts. The bf16 and fp8 jobs are V2's exact
code, byte-identical to V2 (495d347e / 3d26c504). The int8 job is full-map decoder
int8_tensorwise convrot via the comfy-quants stock-ComfyUI producer: 2D decoder
weights (embeddings + attention + dense-MLP; router and encoder stay bf16) plus the
3D MoE expert banks quantized per-expert and restacked ([E, out, in] int8 +
[E, out, 1] per-row scales + num_experts marker). Groupsize per weight: largest of
(256, 64) dividing in_features (gate_up 2816 -> 256; down 704 -> 64; dense mlp.down
2112 -> 64). Requires ComfyUI `diffusion-gemma-finish-dg` containing `a362a5e5`
(packed INT8 ConvRot expert execution). int8 convrot is a torch.matmul, byte-reproducible only within one fixed
environment; the canonical surface is interceptor CPU (--device cpu,
CUDA_VISIBLE_DEVICES=).

The mxfp8_fused job keeps DiffusionGemma's natural fused gate_up bank and quantizes
the 60 MoE banks, 205 decoder-layer matrices, and the tied decoder token embedding
with comfy-quants' deterministic CPU MXFP8 producer. Each weight stores FP8-E4M3
values plus per-32-element UE8M0 scales in CUTLASS/cublas 128x4 blocked layout. The
expert banks use the strict mxfp8_cutlass_fused_moe_v1 marker; activation
microscaling is dynamic at runtime.

The mxfp8_fused_qkv and int8_fused_qkv jobs additionally replace each decoder
attention block's separate Q/K/V (or global Q/K) tensors with one qkv_proj
tensor. The corresponding *_qkv_patch jobs perform only that payload-preserving
structural rewrite on an existing artifact; they do not requantize any value.

Base jobs keep HF naming (model.decoder.*, model.encoder.*), rename the fused expert
banks to <bank>.weight (comfy.ops.MoEExperts), and embed tokenizer.json. QKV jobs also
replace attention projection component keys with qkv_proj. lm_head.* is dropped.

Precisions:
  bf16                          every float tensor -> bfloat16; expert banks -> <bank>.weight.
  fp8                           Kijai FP8 V1 (max_value=416): expert banks (3D) -> float8_e4m3fn
                                with per-expert scale (amax over dims 1,2 / 416); large 2D
                                text-backbone weights (model.decoder.layers.*, max(shape) >= 4096,
                                non-norm) -> float8_e4m3fn per-tensor scale (amax / 416).
                                Everything else -> bfloat16.
  mxfp8_fused                   natural fused gate_up + down 3D expert banks -> float8_e4m3fn
                                and decoder-layer matrices -> float8_e4m3fn with per-32-element
                                UE8M0 scales in CUTLASS 128x4 layout.
  mxfp8_fused_qkv               mxfp8_fused with one concatenated attention projection per block.
  mxfp8_qkv_patch               repack an existing mxfp8_fused artifact without requantization.
  int8_fused_qkv                int8 with one concatenated attention projection per block.
  int8_qkv_patch                repack an existing int8 artifact without requantization.

Examples:
  python convert_diffusion_gemma.py \
      --src ~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/<rev> \
      --job bf16:diffusiongemma_comfy_bf16.safetensors:495d347e1b6c1aa13338741a17d1f5632f3ad4adb11f85f8eeb6ec026db418d1 \
      --job fp8:diffusiongemma_comfy_fp8.safetensors:3d26c504c323bc78fa2d51dbc8433ba4ccf45dcb015b46122d2e37e4c4496015 \
      --job mxfp8_fused:diffusiongemma_comfy_mxfp8_cutlass_fused_moe_v1.safetensors

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

# int8 convrot is delegated to comfy-quants' stock-ComfyUI producer (byte-matches
# ComfyUI's own save path), same as convert_gemma4.py.
from comfy_quants.backends.int8_tensorwise_model_export import _quantize_int8_tensorwise_per_row
from comfy_quants.formats.int8_tensorwise import int8_tensorwise_checkpoint_quant_config
from comfy_quants.formats.mxfp8_blocked import BLOCK_SIZE as MXFP8_BLOCK
from comfy_quants.formats.mxfp8_blocked import quantize_mxfp8_block

FP8_DTYPE = torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8_DTYPE)
FP8_MAX = 416.0  # Kijai DiffusionGemma FP8 V1 convention
EXPERT_HF_SUFFIXES = (".experts.gate_up_proj", ".experts.down_proj")
EXPERT_WEIGHT_SUFFIXES = (".experts.gate_up_proj.weight", ".experts.down_proj.weight")
INT8_VALID_GS = (256, 64)  # convrot Hadamard sizes; prefer largest that divides in_features
MXFP8_FUSED_FORMAT = "mxfp8_cutlass_fused_moe_v1"
MXFP8_FUSED_CONTRACT = "diffusiongemma_mxfp8_cutlass_fused_moe.v1"
MXFP8_NUM_EXPERTS = 128
MXFP8_TIED_EMBEDDING_KEY = "model.decoder.embed_tokens.weight"
MXFP8_TIED_EMBEDDING_SHAPE = (262144, 2816)
MXFP8_FUSED_BANK_SHAPES = {
    ".experts.gate_up_proj.weight": ((128, 1408, 2816), (128, 1408, 88)),
    ".experts.down_proj.weight": ((128, 2816, 704), (128, 2816, 24)),
}
MXFP8_QKV_CONTRACT = "diffusiongemma_mxfp8_qkv_fused.v1"
INT8_QKV_CONTRACT = "diffusiongemma_int8_convrot_qkv_fused.v1"
MXFP8_QKV_HIDDEN_SIZE = 2816
MXFP8_QKV_SCALE_COLS = 88
MXFP8_QKV_LAYER_COUNT = 30
MXFP8_SLIDING_PROJECTIONS = (("q_proj", 4096), ("k_proj", 2048), ("v_proj", 2048))
MXFP8_GLOBAL_PROJECTIONS = (("q_proj", 8192), ("k_proj", 1024))


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


def mxfp8_fused_conf():
    return {
        "format": MXFP8_FUSED_FORMAT,
        "artifact_contract": MXFP8_FUSED_CONTRACT,
        "num_experts": MXFP8_NUM_EXPERTS,
        "group_size": MXFP8_BLOCK,
        "weight_dtype": "float8_e4m3fn",
        "block_scale_dtype": "ue8m0",
        "block_scale_layout": "cutlass_128x4",
        "projection_order": "gate_up",
        "activation_scale": "dynamic_e8m0_1x32",
        "full_precision_matrix_mult": False,
    }


def quantize_mxfp8_fused_bank(k, w):
    expected = next(
        (shapes for suffix, shapes in MXFP8_FUSED_BANK_SHAPES.items() if k.endswith(suffix)),
        None,
    )
    if expected is None:
        raise SystemExit(f"mxfp8_fused: unsupported expert bank {k}")
    expected_weight_shape, expected_scale_shape = expected
    if tuple(w.shape) != expected_weight_shape:
        raise SystemExit(
            f"mxfp8_fused: {k} expected shape {expected_weight_shape}, got {tuple(w.shape)}"
        )
    if w.device.type != "cpu":
        raise SystemExit(f"mxfp8_fused: canonical conversion requires CPU source tensors ({k})")

    qweights = []
    weight_scales = []
    for expert in range(MXFP8_NUM_EXPERTS):
        qweight, weight_scale = quantize_mxfp8_block(w[expert].contiguous())
        qweights.append(qweight)
        weight_scales.append(weight_scale)

    qweight = torch.stack(qweights).contiguous()
    weight_scale = torch.stack(weight_scales).contiguous()
    if qweight.dtype != torch.float8_e4m3fn or tuple(qweight.shape) != expected_weight_shape:
        raise SystemExit(
            f"mxfp8_fused: {k} produced invalid qweight {qweight.dtype} {tuple(qweight.shape)}"
        )
    if weight_scale.dtype != torch.uint8 or tuple(weight_scale.shape) != expected_scale_shape:
        raise SystemExit(
            f"mxfp8_fused: {k} produced invalid scale {weight_scale.dtype} {tuple(weight_scale.shape)}"
        )

    base = k[:-len(".weight")]
    return {
        f"{base}.weight": qweight,
        f"{base}.weight_scale": weight_scale,
        f"{base}.comfy_quant": marker_tensor(mxfp8_fused_conf()),
    }


def mxfp8_eligible_2d(k, v):
    if k == MXFP8_TIED_EMBEDDING_KEY:
        if tuple(v.shape) != MXFP8_TIED_EMBEDDING_SHAPE:
            raise SystemExit(
                f"mxfp8_fused: {k} expected shape {MXFP8_TIED_EMBEDDING_SHAPE}, "
                f"got {tuple(v.shape)}"
            )
        return True
    return (
        k.startswith("model.decoder.layers.")
        and k.endswith(".weight")
        and v.dim() == 2
        and "norm" not in k
        and ".router." not in k
    )


def quantize_mxfp8_2d(k, w):
    if w.device.type != "cpu":
        raise SystemExit(f"mxfp8_fused: canonical conversion requires CPU source tensors ({k})")
    qweight, weight_scale = quantize_mxfp8_block(w.contiguous())
    base = k[:-len(".weight")]
    return {
        f"{base}.weight": qweight.contiguous(),
        f"{base}.weight_scale": weight_scale.contiguous(),
        f"{base}.comfy_quant": marker_tensor(
            {"format": "mxfp8", "full_precision_matrix_mult": False}
        ),
    }


def fuse_mxfp8_attention_projections(sd):
    """Replace DG's separate MXFP8 attention projections without requantization."""
    for layer in range(MXFP8_QKV_LAYER_COUNT):
        projections = (
            MXFP8_GLOBAL_PROJECTIONS if layer % 6 == 5 else MXFP8_SLIDING_PROJECTIONS
        )
        attention = f"model.decoder.layers.{layer}.self_attn"
        fused = f"{attention}.qkv_proj"
        fused_keys = tuple(f"{fused}.{suffix}" for suffix in ("weight", "weight_scale", "comfy_quant"))
        collisions = [key for key in fused_keys if key in sd]
        if collisions:
            raise SystemExit(f"mxfp8 qkv: output keys already exist: {collisions}")
        projection_names = {projection for projection, _ in projections}
        unexpected = [
            f"{attention}.{projection}.{suffix}"
            for projection in ("q_proj", "k_proj", "v_proj")
            if projection not in projection_names
            for suffix in ("weight", "weight_scale", "comfy_quant")
            if f"{attention}.{projection}.{suffix}" in sd
        ]
        if unexpected:
            raise SystemExit(f"mxfp8 qkv: unexpected layer {layer} payload: {unexpected}")

        weights = []
        scales = []
        component_keys = []
        for projection, out_features in projections:
            base = f"{attention}.{projection}"
            weight_key = f"{base}.weight"
            scale_key = f"{base}.weight_scale"
            marker_key = f"{base}.comfy_quant"
            missing = [key for key in (weight_key, scale_key, marker_key) if key not in sd]
            if missing:
                raise SystemExit(f"mxfp8 qkv: missing layer {layer} payload: {missing}")

            weight = sd[weight_key]
            scale = sd[scale_key]
            marker = sd[marker_key]
            expected_weight = (out_features, MXFP8_QKV_HIDDEN_SIZE)
            expected_scale = (out_features, MXFP8_QKV_SCALE_COLS)
            if (
                weight.device.type != "cpu"
                or weight.dtype != torch.float8_e4m3fn
                or tuple(weight.shape) != expected_weight
                or not weight.is_contiguous()
            ):
                raise SystemExit(
                    f"mxfp8 qkv: {weight_key} expected contiguous CPU E4M3 {expected_weight}, "
                    f"got {weight.device} {weight.dtype} {tuple(weight.shape)}"
                )
            if (
                scale.device.type != "cpu"
                or scale.dtype != torch.uint8
                or tuple(scale.shape) != expected_scale
                or not scale.is_contiguous()
            ):
                raise SystemExit(
                    f"mxfp8 qkv: {scale_key} expected contiguous CPU uint8 {expected_scale}, "
                    f"got {scale.device} {scale.dtype} {tuple(scale.shape)}"
                )
            if marker.device.type != "cpu" or marker.dtype != torch.uint8 or marker.ndim != 1:
                raise SystemExit(f"mxfp8 qkv: invalid marker tensor {marker_key}")
            try:
                conf = json.loads(marker.numpy().tobytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SystemExit(f"mxfp8 qkv: invalid marker JSON {marker_key}: {error}") from error
            if conf.get("format") != "mxfp8" or conf.get("full_precision_matrix_mult") is not False:
                raise SystemExit(f"mxfp8 qkv: incompatible marker {marker_key}: {conf}")

            weights.append(weight)
            scales.append(scale)
            component_keys.extend((weight_key, scale_key, marker_key))

        sd[fused_keys[0]] = torch.cat(weights, dim=0).contiguous()
        sd[fused_keys[1]] = torch.cat(scales, dim=0).contiguous()
        sd[fused_keys[2]] = marker_tensor({
            "format": "mxfp8",
            "full_precision_matrix_mult": False,
            "artifact_contract": MXFP8_QKV_CONTRACT,
            "projection_order": [projection for projection, _ in projections],
            "projection_splits": [out_features for _, out_features in projections],
        })
        for key in component_keys:
            del sd[key]

    return MXFP8_QKV_LAYER_COUNT


def fuse_int8_attention_projections(sd):
    """Replace DG's separate INT8 ConvRot projections without requantization."""
    for layer in range(MXFP8_QKV_LAYER_COUNT):
        projections = (
            MXFP8_GLOBAL_PROJECTIONS if layer % 6 == 5 else MXFP8_SLIDING_PROJECTIONS
        )
        attention = f"model.decoder.layers.{layer}.self_attn"
        fused = f"{attention}.qkv_proj"
        suffixes = ("weight", "weight_scale", "comfy_quant")
        fused_keys = tuple(f"{fused}.{suffix}" for suffix in suffixes)
        collisions = [key for key in fused_keys if key in sd]
        if collisions:
            raise SystemExit(f"int8 qkv: output keys already exist: {collisions}")

        projection_names = {projection for projection, _ in projections}
        unexpected = [
            f"{attention}.{projection}.{suffix}"
            for projection in ("q_proj", "k_proj", "v_proj")
            if projection not in projection_names
            for suffix in suffixes
            if f"{attention}.{projection}.{suffix}" in sd
        ]
        if unexpected:
            raise SystemExit(f"int8 qkv: unexpected layer {layer} payload: {unexpected}")

        weights = []
        scales = []
        component_keys = []
        for projection, out_features in projections:
            base = f"{attention}.{projection}"
            weight_key, scale_key, marker_key = (f"{base}.{suffix}" for suffix in suffixes)
            missing = [key for key in (weight_key, scale_key, marker_key) if key not in sd]
            if missing:
                raise SystemExit(f"int8 qkv: missing layer {layer} payload: {missing}")

            weight = sd[weight_key]
            scale = sd[scale_key]
            marker = sd[marker_key]
            expected_weight = (out_features, MXFP8_QKV_HIDDEN_SIZE)
            expected_scale = (out_features, 1)
            if (
                weight.device.type != "cpu"
                or weight.dtype != torch.int8
                or tuple(weight.shape) != expected_weight
                or not weight.is_contiguous()
            ):
                raise SystemExit(f"int8 qkv: invalid weight {weight_key}: {weight.dtype} {tuple(weight.shape)}")
            if (
                scale.device.type != "cpu"
                or scale.dtype != torch.float32
                or tuple(scale.shape) != expected_scale
                or not scale.is_contiguous()
            ):
                raise SystemExit(f"int8 qkv: invalid scale {scale_key}: {scale.dtype} {tuple(scale.shape)}")
            if marker.device.type != "cpu" or marker.dtype != torch.uint8 or marker.ndim != 1:
                raise SystemExit(f"int8 qkv: invalid marker tensor {marker_key}")
            try:
                conf = json.loads(marker.numpy().tobytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SystemExit(f"int8 qkv: invalid marker JSON {marker_key}: {error}") from error
            expected_conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
            if any(conf.get(key) != value for key, value in expected_conf.items()):
                raise SystemExit(f"int8 qkv: incompatible marker {marker_key}: {conf}")

            weights.append(weight)
            scales.append(scale)
            component_keys.extend((weight_key, scale_key, marker_key))

        sd[fused_keys[0]] = torch.cat(weights, dim=0).contiguous()
        sd[fused_keys[1]] = torch.cat(scales, dim=0).contiguous()
        sd[fused_keys[2]] = marker_tensor({
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
            "artifact_contract": INT8_QKV_CONTRACT,
            "projection_order": [projection for projection, _ in projections],
            "projection_splits": [out_features for _, out_features in projections],
        })
        for key in component_keys:
            del sd[key]

    return MXFP8_QKV_LAYER_COUNT


def int8_groupsize(in_features):
    return next((g for g in INT8_VALID_GS if in_features % g == 0), None)


def int8_convrot_eligible_2d(k, v):
    # Full-map decoder policy (mirrors convert_gemma4.py): embeddings + attention +
    # dense-MLP linears; router/control weights and the encoder stay bf16.
    if not (k.startswith("model.decoder.") and k.endswith(".weight") and v.dim() == 2):
        return False
    if "norm" in k or ".router." in k:
        return False
    return int8_groupsize(v.shape[1]) is not None


def quantize_int8_2d(k, w, dev):
    base = k[:-len(".weight")]
    gs = int8_groupsize(w.shape[1])
    qweight, scale, rotated = _quantize_int8_tensorwise_per_row(
        w.contiguous().to(dev), convrot=True, group_size=gs)
    if not rotated:
        raise SystemExit(f"int8: producer refused convrot rotation for {k} (gs {gs})")
    marker = int8_tensorwise_checkpoint_quant_config(convrot=True, convrot_groupsize=gs)
    return {
        f"{base}.weight": qweight.detach().to("cpu").contiguous(),
        f"{base}.weight_scale": scale.detach().to("cpu").contiguous(),
        f"{base}.comfy_quant": marker_tensor(marker),
    }


def quantize_int8_bank(k, w, dev):
    base = k[:-len(".weight")]
    num_experts, _, in_features = w.shape
    gs = int8_groupsize(in_features)
    if gs is None:
        raise SystemExit(f"int8: no valid convrot groupsize for bank {k} (in_features {in_features})")
    qs, ss = [], []
    for e in range(num_experts):
        qweight, scale, rotated = _quantize_int8_tensorwise_per_row(
            w[e].contiguous().to(dev), convrot=True, group_size=gs)
        if not rotated:
            raise SystemExit(f"int8: producer refused convrot rotation for {k} expert {e} (gs {gs})")
        qs.append(qweight.detach().to("cpu"))
        ss.append(scale.detach().to("cpu"))
    marker = int8_tensorwise_checkpoint_quant_config(convrot=True, convrot_groupsize=gs)
    marker["num_experts"] = num_experts
    return {
        f"{base}.weight": torch.stack(qs).contiguous(),
        f"{base}.weight_scale": torch.stack(ss).contiguous(),
        f"{base}.comfy_quant": marker_tensor(marker),
    }


def cast(sd, precision, device="cpu"):
    if precision == "mxfp8_qkv_patch":
        out = dict(sd)
        fused = fuse_mxfp8_attention_projections(out)
        print(f"{precision}: fused {fused} attention projection groups without requantization")
        return out
    if precision == "int8_qkv_patch":
        out = dict(sd)
        fused = fuse_int8_attention_projections(out)
        print(f"{precision}: fused {fused} attention projection groups without requantization")
        return out

    out = {}
    nq = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if precision == "bf16":
            out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        elif precision == "fp8":
            if is_expert_bank(k):
                out.update(quantize_bank(k, v))
                nq += 1
            elif should_quantize_2d(k, v):
                out.update(quantize_2d(k, v))
                nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        elif precision in ("mxfp8_fused", "mxfp8_fused_qkv"):
            if is_expert_bank(k):
                out.update(quantize_mxfp8_fused_bank(k, v))
                nq += 1
            elif mxfp8_eligible_2d(k, v):
                out.update(quantize_mxfp8_2d(k, v))
                nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        elif precision in ("int8", "int8_fused_qkv"):
            if is_expert_bank(k):
                out.update(quantize_int8_bank(k, v, device))
                nq += 1
            elif int8_convrot_eligible_2d(k, v):
                out.update(quantize_int8_2d(k, v, device))
                nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        else:
            raise SystemExit(f"unknown precision: {precision}")
    if precision == "mxfp8_fused_qkv":
        fused = fuse_mxfp8_attention_projections(out)
        print(f"{precision}: fused {fused} attention projection groups")
    if precision == "int8_fused_qkv":
        fused = fuse_int8_attention_projections(out)
        print(f"{precision}: fused {fused} attention projection groups")
    if precision in ("fp8", "mxfp8_fused", "mxfp8_fused_qkv", "int8", "int8_fused_qkv"):
        print(f"{precision}: quantized {nq} weights")
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert DiffusionGemma to ComfyUI safetensors (V2, Kijai conversion).")
    ap.add_argument("--src", required=True, help="HF snapshot dir or a ComfyUI bf16 safetensors")
    ap.add_argument("--job", action="append", required=True, metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; one source load serves every job")
    ap.add_argument("--device", default="cpu",
                    help="int8 convrot device; cpu is the canonical byte-reproducible target "
                         "(interceptor CPU). bf16/fp8/MXFP8 jobs ignore this; "
                         "MXFP8 conversion always uses the deterministic CPU producer.")
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
        tensors = cast(base, precision, device=args.device)
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
