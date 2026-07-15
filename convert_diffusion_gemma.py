#!/usr/bin/env python3
"""Create the six canonical ComfyUI DiffusionGemma-26B-A4B artifacts.

Source: google/diffusiongemma-26B-A4B-it revision 0f28bc42. Each public job is the
single accepted recipe for its quant and must match the fixed SHA256 below:

  bf16   BF16 mapped checkpoint                                      495d347e...
  fp8    Kijai scaled FP8                                            3d26c504...
  int8   full-map CPU ConvRot W8A8 with fused QKV                    1cdeb5de...
  int4   routed-expert CPU ConvRot W4A8, remaining linears W8A8      7d95437c...
  mxfp8  fused experts, decoder linears, tied embedding, fused QKV    211f3140...
  nvfp4  rev-0 split experts and 205 decoder linears                 a07a8cda...

There are no public patch, intermediate, or alternate-layout jobs. Structural
fusion is part of the canonical INT8 and MXFP8 recipes. The accepted SHA is not a
caller option: every conversion verifies it and fails on any byte drift.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# int8 convrot is delegated to comfy-quants' stock-ComfyUI producer (byte-matches
# ComfyUI's own save path), same as convert_gemma4.py.
try:
    from comfy_quants.backends.convrot_w4a4_model_export import _quantize_convrot_w4a4_per_row
    from comfy_quants.backends.int8_tensorwise_model_export import _quantize_int8_tensorwise_per_row
    from comfy_quants.formats.convrot_w4a4 import convrot_w4a4_checkpoint_quant_config
    from comfy_quants.formats.int8_tensorwise import int8_tensorwise_checkpoint_quant_config
    from comfy_quants.formats.mxfp8_blocked import BLOCK_SIZE as MXFP8_BLOCK
    from comfy_quants.formats.mxfp8_blocked import quantize_mxfp8_block
except ModuleNotFoundError as error:
    if error.name != "comfy_quants":
        raise
    _quantize_convrot_w4a4_per_row = None
    _quantize_int8_tensorwise_per_row = None
    convrot_w4a4_checkpoint_quant_config = None
    int8_tensorwise_checkpoint_quant_config = None
    quantize_mxfp8_block = None
    MXFP8_BLOCK = 32

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
CANONICAL_SHA256 = {
    "bf16": "495d347e1b6c1aa13338741a17d1f5632f3ad4adb11f85f8eeb6ec026db418d1",
    "fp8": "3d26c504c323bc78fa2d51dbc8433ba4ccf45dcb015b46122d2e37e4c4496015",
    "int8": "1cdeb5deb7f553257c06a54dffc47be2b3242781808319840074e1bcbfe48401",
    "int4": "7d95437cf72720302d672c70695251d6c17f5c755f1e7842436db62d1459f881",
    "mxfp8": "211f31404bc1ad56a912912dac3caacb0e420a0c0d31aba30a537e3c3370bce7",
    "nvfp4": "a07a8cdacd46fb106a718bf07a628645f078c306eb7e2097dfcff4b5f7677cdc",
}
NVFP4_ATTN_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
)
NVFP4_MLP_SUFFIXES = (
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def marker_tensor(conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def compact_marker_tensor(conf):
    data = json.dumps(conf, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8)


def sorted_marker_tensor(conf):
    data = json.dumps(conf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8)


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


def nvfp4_layout():
    comfy_root = Path(os.environ.get("COMFYUI_ROOT", Path.home() / "dev_master" / "ComfyUI"))
    sys.path.insert(0, str(comfy_root))
    try:
        from comfy.quant_ops import TensorCoreNVFP4Layout
    except ModuleNotFoundError as error:
        raise SystemExit(f"nvfp4 conversion requires ComfyUI at {comfy_root}: {error}") from error
    return TensorCoreNVFP4Layout


def nvfp4_should_quantize_2d(k, v):
    return (
        k.startswith("model.decoder.layers.")
        and k.endswith(NVFP4_ATTN_SUFFIXES + NVFP4_MLP_SUFFIXES)
        and v.dim() == 2
    )


def quantize_nvfp4_2d(w, device):
    layout = nvfp4_layout()
    x = w.to(device=device, dtype=torch.bfloat16)
    qdata, params = layout.quantize(x)
    out = {
        suffix: value.cpu()
        for suffix, value in layout.state_dict_tensors(qdata, params).items()
    }
    del x, qdata, params
    return out


def add_nvfp4_2d(out, k, w, device):
    base = k[:-len(".weight")]
    for suffix, value in quantize_nvfp4_2d(w, device).items():
        out[f"{base}.weight{suffix}"] = value
    out[f"{base}.comfy_quant"] = sorted_marker_tensor(
        {"format": "nvfp4", "full_precision_matrix_mult": False}
    )


def add_nvfp4_bank(out, base, bank, device):
    q_chunks = []
    scale_chunks = collections.defaultdict(list)
    for expert in range(bank.shape[0]):
        tensors = quantize_nvfp4_2d(bank[expert], device)
        q_chunks.append(tensors.pop(""))
        for suffix, value in tensors.items():
            scale_chunks[suffix].append(value)
    out[f"{base}.weight"] = torch.stack(q_chunks, dim=0)
    for suffix, chunks in scale_chunks.items():
        out[f"{base}.weight{suffix}"] = torch.stack(chunks, dim=0)
    out[f"{base}.comfy_quant"] = sorted_marker_tensor(
        {
            "format": "nvfp4",
            "full_precision_matrix_mult": False,
            "num_experts": int(bank.shape[0]),
        }
    )


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
    if quantize_mxfp8_block is None:
        raise SystemExit("mxfp8 conversion requires comfy-quants")
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
    if quantize_mxfp8_block is None:
        raise SystemExit("mxfp8 conversion requires comfy-quants")
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


def int4_convrot_eligible_2d(k, v):
    return k != MXFP8_TIED_EMBEDDING_KEY and int8_convrot_eligible_2d(k, v)


def quantize_int8_2d(k, w, dev):
    if _quantize_int8_tensorwise_per_row is None:
        raise SystemExit("int8 conversion requires comfy-quants")
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
    if _quantize_int8_tensorwise_per_row is None:
        raise SystemExit("int8 conversion requires comfy-quants")
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


def quantize_int4_2d(k, w, dev):
    if _quantize_convrot_w4a4_per_row is None:
        raise SystemExit("int4 conversion requires comfy-quants with ConvRot W4A4 support")
    base = k[:-len(".weight")]
    gs = int8_groupsize(w.shape[1])
    qweight, scale = _quantize_convrot_w4a4_per_row(
        w.contiguous().to(dev), group_size=gs)
    marker = convrot_w4a4_checkpoint_quant_config(
        convrot_groupsize=gs,
        linear_dtype="int8",
    )
    return {
        f"{base}.weight": qweight.detach().to("cpu").contiguous(),
        f"{base}.weight_scale": scale.detach().to("cpu").contiguous(),
        f"{base}.comfy_quant": marker_tensor(marker),
    }


def quantize_int4_bank(k, w, dev):
    if _quantize_convrot_w4a4_per_row is None:
        raise SystemExit("int4 conversion requires comfy-quants with ConvRot W4A4 support")
    base = k[:-len(".weight")]
    num_experts, _, in_features = w.shape
    gs = int8_groupsize(in_features)
    if gs is None:
        raise SystemExit(f"int4: no valid convrot groupsize for bank {k} (in_features {in_features})")
    quantized = [
        _quantize_convrot_w4a4_per_row(w[e].contiguous().to(dev), group_size=gs)
        for e in range(num_experts)
    ]
    marker = convrot_w4a4_checkpoint_quant_config(convrot_groupsize=gs)
    marker["num_experts"] = num_experts
    marker["linear_dtype"] = "int8"
    return {
        f"{base}.weight": torch.stack([item[0].detach().to("cpu") for item in quantized]).contiguous(),
        f"{base}.weight_scale": torch.stack([item[1].detach().to("cpu") for item in quantized]).contiguous(),
        f"{base}.comfy_quant": compact_marker_tensor(marker),
    }


def cast(sd, precision):
    out = {}
    nq = 0
    cpu = "cpu"
    nvfp4_device = torch.device("cuda:0") if precision == "nvfp4" else None
    if precision == "nvfp4" and not torch.cuda.is_available():
        raise SystemExit("canonical nvfp4 conversion requires CUDA device 0")
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
        elif precision == "mxfp8":
            if is_expert_bank(k):
                out.update(quantize_mxfp8_fused_bank(k, v))
                nq += 1
            elif mxfp8_eligible_2d(k, v):
                out.update(quantize_mxfp8_2d(k, v))
                nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        elif precision == "int8":
            if is_expert_bank(k):
                out.update(quantize_int8_bank(k, v, cpu))
                nq += 1
            elif int8_convrot_eligible_2d(k, v):
                out.update(quantize_int8_2d(k, v, cpu))
                nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        elif precision == "int4":
            if is_expert_bank(k):
                out.update(quantize_int4_bank(k, v, cpu))
                nq += 1
            elif int8_convrot_eligible_2d(k, v):
                out.update(quantize_int8_2d(k, v, cpu))
                nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        elif precision == "nvfp4":
            if k.endswith(".experts.gate_up_proj.weight"):
                half = v.shape[1] // 2
                if v.shape[1] % 2:
                    raise SystemExit(f"{k}: odd fused gate_up dimension {v.shape[1]}")
                prefix = k[:-len(".experts.gate_up_proj.weight")]
                add_nvfp4_bank(out, f"{prefix}.experts.gate_proj", v[:, :half, :], nvfp4_device)
                add_nvfp4_bank(out, f"{prefix}.experts.up_proj", v[:, half:, :], nvfp4_device)
                nq += 2
            elif k.endswith(".experts.down_proj.weight"):
                add_nvfp4_bank(out, k[:-len(".weight")], v, nvfp4_device)
                nq += 1
            elif nvfp4_should_quantize_2d(k, v):
                add_nvfp4_2d(out, k, v, nvfp4_device)
                nq += 1
            else:
                out[k] = v.to(torch.bfloat16) if (k != "tokenizer_json" and v.is_floating_point()) else v
        else:
            raise SystemExit(f"unknown precision: {precision}")
    if precision == "mxfp8":
        fused = fuse_mxfp8_attention_projections(out)
        print(f"{precision}: fused {fused} attention projection groups")
    if precision == "int8":
        fused = fuse_int8_attention_projections(out)
        print(f"{precision}: fused {fused} attention projection groups")
    if precision != "bf16":
        print(f"{precision}: quantized {nq} weights")
    return out


def main():
    ap = argparse.ArgumentParser(description="Create canonical ComfyUI DiffusionGemma artifacts.")
    ap.add_argument("--src", required=True, help="HF snapshot dir or a ComfyUI bf16 safetensors")
    ap.add_argument(
        "--job",
        action="append",
        required=True,
        metavar="QUANT:OUT",
        help="repeatable canonical job: bf16, fp8, int8, int4, mxfp8, or nvfp4",
    )
    args = ap.parse_args()

    base = load_base(args.src)
    for job in args.job:
        precision, sep, out = job.partition(":")
        if not sep or not out or precision not in CANONICAL_SHA256:
            choices = ", ".join(CANONICAL_SHA256)
            raise SystemExit(f"invalid --job {job!r}; expected one of {choices} as QUANT:OUT")
        if ":" in out:
            raise SystemExit("the canonical SHA is fixed by the converter, not supplied by the caller")
        tensors = cast(base, precision)
        save_file(tensors, out)
        digest = sha256(out)
        expected = CANONICAL_SHA256[precision]
        verdict = "OK" if digest == expected else "MISMATCH"
        print(f"{precision:7s} {digest}  {out}  {verdict}")
        if digest != expected:
            raise SystemExit(f"{precision} SHA256 mismatch: expected {expected}, got {digest}")


if __name__ == "__main__":
    main()
