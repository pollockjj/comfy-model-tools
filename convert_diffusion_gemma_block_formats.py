#!/usr/bin/env python3
"""Working-first DiffusionGemma block-format converter.

The source is an existing ComfyUI BF16 DiffusionGemma text encoder. Coverage
matches the useful FP8V2 policy for decoder compute weights. The legacy MXFP8
and NVFP4 paths quantize split expert banks. The fused NVFP4 path transcodes
NVIDIA's calibrated expert payload without requantizing it and leaves all
non-expert tensors in BF16.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", Path.home() / "dev_master" / "ComfyUI"))
sys.path.insert(0, str(COMFY_ROOT))

from comfy.quant_ops import TensorCoreMXFP8Layout, TensorCoreNVFP4Layout  # noqa: E402


SOURCE_DEFAULT = COMFY_ROOT / "models/text_encoders/diffusiongemma_comfy_bf16.safetensors"
NVIDIA_DEFAULT = Path.home() / "dev_servant" / "diffusiongemma_nvfp4"
OUT_DEFAULTS = {
    "mxfp8": COMFY_ROOT / "models/text_encoders/diffusiongemma_comfy_mxfp8_working.safetensors",
    "nvfp4": COMFY_ROOT / "models/text_encoders/diffusiongemma_comfy_nvfp4_working.safetensors",
    "nvfp4_fused": COMFY_ROOT / "models/text_encoders/diffusiongemma_comfy_nvfp4_cutlass_fused_moe_v1.safetensors",
}
MANIFEST_DEFAULT = Path(__file__).resolve().with_name("conversion_manifest.json")

GATE_UP_EXPERT = ".experts.gate_up_proj.weight"
DOWN_EXPERT = ".experts.down_proj.weight"
ATTN_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
)
MLP_SUFFIXES = (
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)
FORMAT_LAYOUTS = {
    "mxfp8": TensorCoreMXFP8Layout,
    "nvfp4": TensorCoreNVFP4Layout,
}
FORMAT_CHOICES = (*FORMAT_LAYOUTS, "nvfp4_fused")
FUSED_NVFP4_FORMAT = "nvfp4_cutlass_fused_moe_v1"
FUSED_NVFP4_CONTRACT = "diffusiongemma_nvfp4_cutlass_fused_moe.v1"
EXPERT_LAYER_RE = re.compile(r"^model\.decoder\.layers\.(\d+)\.experts\.")
OFFICIAL_EXPERT_SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")
NUM_EXPERTS = 128


class NvidiaTensorReader:
    """Index-backed reader for an official sharded NVIDIA checkpoint."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.index_path = model_dir / "model.safetensors.index.json"
        if not self.index_path.is_file():
            raise SystemExit(f"missing NVIDIA index: {self.index_path}")
        index = json.loads(self.index_path.read_text())
        self.weight_map = index.get("weight_map")
        if not isinstance(self.weight_map, dict):
            raise SystemExit(f"invalid NVIDIA index weight_map: {self.index_path}")
        self._stack = contextlib.ExitStack()
        self._shards = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stack.close()

    def get_tensor(self, key: str) -> torch.Tensor:
        shard_name = self.weight_map.get(key)
        if shard_name is None:
            raise SystemExit(f"NVIDIA checkpoint is missing {key}")
        handle = self._shards.get(shard_name)
        if handle is None:
            shard_path = self.model_dir / shard_name
            if not shard_path.is_file():
                raise SystemExit(f"missing NVIDIA shard: {shard_path}")
            handle = self._stack.enter_context(safe_open(shard_path, framework="pt"))
            self._shards[shard_name] = handle
        return handle.get_tensor(key)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def quant_tensor(conf: dict) -> torch.Tensor:
    data = json.dumps(conf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8)


def fused_nvfp4_conf() -> dict:
    return {
        "format": FUSED_NVFP4_FORMAT,
        "artifact_contract": FUSED_NVFP4_CONTRACT,
        "num_experts": NUM_EXPERTS,
        "group_size": 16,
        "nibble_order": "low_first",
        "block_scale_layout": "cutlass_128x4",
        "projection_order": "up_gate",
        "activation_scale": "static",
        "full_precision_matrix_mult": False,
    }


def layer_from_expert_key(key: str) -> int:
    match = EXPERT_LAYER_RE.match(key)
    if match is None:
        raise SystemExit(f"cannot derive decoder layer from {key}")
    return int(match.group(1))


def exact_common_scale(values: torch.Tensor, label: str) -> torch.Tensor:
    if values.shape != (NUM_EXPERTS,):
        raise SystemExit(f"{label}: expected {NUM_EXPERTS} scalar scales, got {tuple(values.shape)}")
    if not torch.equal(values, values[0].expand_as(values)):
        mismatches = int(torch.count_nonzero(values != values[0]))
        raise SystemExit(f"{label}: {mismatches} expert scales differ")
    return values[0].clone()


def swizzle_cutlass_block_scale(scale: torch.Tensor, label: str) -> torch.Tensor:
    if scale.dtype != torch.float8_e4m3fn or scale.dim() != 3:
        raise SystemExit(f"{label}: expected 3D F8_E4M3 block scale, got {scale.dtype} {tuple(scale.shape)}")
    experts, output_features, scale_columns = scale.shape
    if experts != NUM_EXPERTS or output_features % 128 or scale_columns % 4:
        raise SystemExit(f"{label}: unsupported CUTLASS scale shape {tuple(scale.shape)}")
    payload = scale.contiguous().view(torch.uint8)
    payload = (
        payload.reshape(experts, output_features // 128, 4, 32, scale_columns // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(experts, output_features, scale_columns)
    )
    return payload.view(torch.float8_e4m3fn)


def read_official_projection(reader: NvidiaTensorReader, layer: int, projection: str) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
    for expert in range(NUM_EXPERTS):
        base = f"model.decoder.layers.{layer}.experts.{expert}.{projection}_proj"
        for suffix in OFFICIAL_EXPERT_SUFFIXES:
            chunks[suffix].append(reader.get_tensor(f"{base}.{suffix}"))
    return {suffix: torch.stack(values) for suffix, values in chunks.items()}


def add_official_fused_gate_up(
    out: dict[str, torch.Tensor], prefix: str, layer: int, reader: NvidiaTensorReader
) -> None:
    up = read_official_projection(reader, layer, "up")
    gate = read_official_projection(reader, layer, "gate")

    if not torch.equal(up["weight_scale_2"], gate["weight_scale_2"]):
        mismatches = int(torch.count_nonzero(up["weight_scale_2"] != gate["weight_scale_2"]))
        raise SystemExit(f"layer {layer}: {mismatches} gate/up global scales differ")
    if not torch.equal(up["input_scale"], gate["input_scale"]):
        mismatches = int(torch.count_nonzero(up["input_scale"] != gate["input_scale"]))
        raise SystemExit(f"layer {layer}: {mismatches} gate/up input scales differ")

    base = f"{prefix}.experts.gate_up_proj"
    qdata = torch.cat((up["weight"], gate["weight"]), dim=1)
    logical_scale = torch.cat((up["weight_scale"], gate["weight_scale"]), dim=1)
    expected_q_shape = (NUM_EXPERTS, 1408, 1408)
    expected_scale_shape = (NUM_EXPERTS, 1408, 176)
    if qdata.shape != expected_q_shape or logical_scale.shape != expected_scale_shape:
        raise SystemExit(
            f"layer {layer}: bad fused gate/up shapes {tuple(qdata.shape)} {tuple(logical_scale.shape)}"
        )
    out[f"{base}.weight"] = qdata
    out[f"{base}.weight_scale"] = swizzle_cutlass_block_scale(logical_scale, f"layer {layer} gate/up")
    out[f"{base}.weight_scale_2"] = up["weight_scale_2"]
    out[f"{base}.input_scale"] = exact_common_scale(up["input_scale"], f"layer {layer} gate/up input")
    out[f"{base}.comfy_quant"] = quant_tensor(fused_nvfp4_conf())


def add_official_down(out: dict[str, torch.Tensor], prefix: str, layer: int, reader: NvidiaTensorReader) -> None:
    down = read_official_projection(reader, layer, "down")
    expected_q_shape = (NUM_EXPERTS, 2816, 352)
    expected_scale_shape = (NUM_EXPERTS, 2816, 44)
    if down["weight"].shape != expected_q_shape or down["weight_scale"].shape != expected_scale_shape:
        raise SystemExit(
            f"layer {layer}: bad down shapes {tuple(down['weight'].shape)} {tuple(down['weight_scale'].shape)}"
        )
    base = f"{prefix}.experts.down_proj"
    out[f"{base}.weight"] = down["weight"]
    out[f"{base}.weight_scale"] = swizzle_cutlass_block_scale(down["weight_scale"], f"layer {layer} down")
    out[f"{base}.weight_scale_2"] = down["weight_scale_2"]
    out[f"{base}.input_scale"] = exact_common_scale(down["input_scale"], f"layer {layer} down input")
    out[f"{base}.comfy_quant"] = quant_tensor(fused_nvfp4_conf())


def should_quantize_2d(key: str, tensor) -> bool:
    return (
        key.startswith("model.decoder.layers.")
        and key.endswith(ATTN_SUFFIXES + MLP_SUFFIXES)
        and tensor.dim() == 2
    )


def quantize_2d(fmt: str, tensor: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    layout = FORMAT_LAYOUTS[fmt]
    x = tensor.to(device=device, dtype=torch.bfloat16)
    qdata, params = layout.quantize(x)
    out = {}
    for suffix, value in layout.state_dict_tensors(qdata, params).items():
        value = value.cpu()
        if fmt == "mxfp8" and suffix == "_scale":
            # Comfy's custom safetensors loader views uint8 bytes back to E8M0.
            # Native F8_E8M0 appears in safetensors headers but is not in its
            # loader dtype map yet.
            value = value.view(torch.uint8)
        out[suffix] = value
    del x, qdata, params
    return out


def add_quantized_2d(out: dict[str, torch.Tensor], key: str, fmt: str, tensor: torch.Tensor, device: torch.device) -> None:
    base = key[:-len(".weight")]
    for suffix, value in quantize_2d(fmt, tensor, device).items():
        out[f"{base}.weight{suffix}"] = value
    out[f"{base}.comfy_quant"] = quant_tensor({"format": fmt, "full_precision_matrix_mult": False})


def quantize_expert_bank(fmt: str, bank: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    q_chunks: list[torch.Tensor] = []
    scale_chunks: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
    for i in range(bank.shape[0]):
        tensors = quantize_2d(fmt, bank[i], device)
        q_chunks.append(tensors.pop(""))
        for suffix, value in tensors.items():
            scale_chunks[suffix].append(value)
    out = {"": torch.stack(q_chunks, dim=0)}
    for suffix, chunks in scale_chunks.items():
        out[suffix] = torch.stack(chunks, dim=0)
    return out


def add_quantized_bank(out: dict[str, torch.Tensor], base: str, fmt: str, bank: torch.Tensor, device: torch.device) -> None:
    for suffix, value in quantize_expert_bank(fmt, bank, device).items():
        out[f"{base}.weight{suffix}"] = value
    out[f"{base}.comfy_quant"] = quant_tensor(
        {"format": fmt, "full_precision_matrix_mult": False, "num_experts": int(bank.shape[0])}
    )


def floating_passthrough(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(tensor) and tensor.dtype != torch.bfloat16:
        return tensor.to(torch.bfloat16)
    return tensor


def convert_one(
    src: Path,
    out_path: Path,
    fmt: str,
    device: torch.device,
    nvidia_model_dir: Path,
) -> dict:
    if fmt not in FORMAT_CHOICES:
        raise SystemExit(f"unsupported format: {fmt}")
    stats = collections.Counter()
    sd_new: dict[str, torch.Tensor] = {}
    fused_nvfp4 = fmt == "nvfp4_fused"

    with contextlib.ExitStack() as stack:
        f = stack.enter_context(safe_open(src, framework="pt"))
        official = stack.enter_context(NvidiaTensorReader(nvidia_model_dir)) if fused_nvfp4 else None
        keys = list(f.keys())
        for index, key in enumerate(keys, start=1):
            if fused_nvfp4 and key.endswith(GATE_UP_EXPERT):
                prefix = key[:-len(GATE_UP_EXPERT)]
                layer = layer_from_expert_key(key)
                add_official_fused_gate_up(sd_new, prefix, layer, official)
                stats["official_fused_gate_up_source"] += 1
                stats["expert_banks_written"] += 1
                print(f"{fmt}: transcoded layer {layer} fused gate/up")
                continue
            if fused_nvfp4 and key.endswith(DOWN_EXPERT):
                prefix = key[:-len(DOWN_EXPERT)]
                layer = layer_from_expert_key(key)
                add_official_down(sd_new, prefix, layer, official)
                stats["official_down_source"] += 1
                stats["expert_banks_written"] += 1
                print(f"{fmt}: transcoded layer {layer} down")
                continue

            tensor = f.get_tensor(key)
            if key.endswith(GATE_UP_EXPERT):
                half = tensor.shape[1] // 2
                if tensor.shape[1] % 2:
                    raise SystemExit(f"{key}: odd fused gate_up dimension {tensor.shape[1]}")
                prefix = key[:-len(GATE_UP_EXPERT)]
                add_quantized_bank(sd_new, f"{prefix}.experts.gate_proj", fmt, tensor[:, :half, :], device)
                add_quantized_bank(sd_new, f"{prefix}.experts.up_proj", fmt, tensor[:, half:, :], device)
                stats["expert_gate_up_split_source"] += 1
                stats["expert_banks_written"] += 2
            elif key.endswith(DOWN_EXPERT):
                base = key[:-len(".weight")]
                add_quantized_bank(sd_new, base, fmt, tensor, device)
                stats["expert_down_source"] += 1
                stats["expert_banks_written"] += 1
            elif not fused_nvfp4 and should_quantize_2d(key, tensor):
                add_quantized_2d(sd_new, key, fmt, tensor, device)
                stats["decoder_2d_quantized"] += 1
            elif key.endswith(".experts.gate_up_proj.weight") or key.endswith(".experts.down_proj.weight"):
                raise SystemExit(f"{key}: expert key escaped quantization policy")
            else:
                sd_new[key] = floating_passthrough(tensor)
                stats["passthrough"] += 1

            del tensor
            if index % 50 == 0:
                torch.cuda.empty_cache()
                print(f"{fmt}: processed {index}/{len(keys)} tensors")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd_new, out_path)
    digest = sha256(out_path)
    dtype_counts = collections.Counter(str(t.dtype) for t in sd_new.values())
    size_bytes = out_path.stat().st_size
    manifest = {
        "format": fmt,
        "source_path": str(src),
        "output_path": str(out_path),
        "sha256": digest,
        "bytes": size_bytes,
        "tensor_count": len(sd_new),
        "stats": dict(sorted(stats.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "coverage": {
            "expert_representation": "fused_calibrated_official" if fused_nvfp4 else "split_unfused",
            "expert_banks_written": int(stats["expert_banks_written"]),
            "decoder_2d_quantized": int(stats["decoder_2d_quantized"]),
        },
    }
    if fused_nvfp4:
        manifest["nvidia_model_dir"] = str(nvidia_model_dir)
        manifest["nvidia_index_sha256"] = sha256(nvidia_model_dir / "model.safetensors.index.json")
        manifest["artifact_contract"] = FUSED_NVFP4_CONTRACT
    del sd_new
    torch.cuda.empty_cache()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--format", choices=sorted(FORMAT_CHOICES), action="append", required=True)
    parser.add_argument("--nvidia-model-dir", type=Path, default=NVIDIA_DEFAULT)
    parser.add_argument("--out-mxfp8", type=Path, default=OUT_DEFAULTS["mxfp8"])
    parser.add_argument("--out-nvfp4", type=Path, default=OUT_DEFAULTS["nvfp4"])
    parser.add_argument("--out-nvfp4-fused", type=Path, default=OUT_DEFAULTS["nvfp4_fused"])
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    args = parser.parse_args()

    needs_cuda = any(fmt in FORMAT_LAYOUTS for fmt in args.format)
    if needs_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this converter")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    outputs = {
        "mxfp8": args.out_mxfp8,
        "nvfp4": args.out_nvfp4,
        "nvfp4_fused": args.out_nvfp4_fused,
    }

    manifests = []
    for fmt in args.format:
        print(f"converting {fmt} -> {outputs[fmt]}")
        manifests.append(convert_one(args.src, outputs[fmt], fmt, device, args.nvidia_model_dir))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"jobs": manifests}, indent=2, sort_keys=True) + "\n")
    print(args.manifest)


if __name__ == "__main__":
    main()
