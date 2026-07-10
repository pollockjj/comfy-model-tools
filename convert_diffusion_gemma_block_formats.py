#!/usr/bin/env python3
"""Working-first DiffusionGemma MXFP8/NVFP4 converter.

The source is an existing ComfyUI BF16 DiffusionGemma text encoder. Coverage
matches the useful FP8V2 policy for decoder compute weights, with expert banks
emitted in the split/unfused form so DiffusionGemma bypasses grouped bank bmm.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", Path.home() / "dev_master" / "ComfyUI"))
sys.path.insert(0, str(COMFY_ROOT))

from comfy.quant_ops import TensorCoreMXFP8Layout, TensorCoreNVFP4Layout  # noqa: E402


SOURCE_DEFAULT = COMFY_ROOT / "models/text_encoders/diffusiongemma_comfy_bf16.safetensors"
OUT_DEFAULTS = {
    "mxfp8": COMFY_ROOT / "models/text_encoders/diffusiongemma_comfy_mxfp8_working.safetensors",
    "nvfp4": COMFY_ROOT / "models/text_encoders/diffusiongemma_comfy_nvfp4_working.safetensors",
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def quant_tensor(conf: dict) -> torch.Tensor:
    data = json.dumps(conf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8)


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


def convert_one(src: Path, out_path: Path, fmt: str, device: torch.device) -> dict:
    if fmt not in FORMAT_LAYOUTS:
        raise SystemExit(f"unsupported format: {fmt}")
    stats = collections.Counter()
    sd_new: dict[str, torch.Tensor] = {}

    with safe_open(src, framework="pt") as f:
        keys = list(f.keys())
        for index, key in enumerate(keys, start=1):
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
            elif should_quantize_2d(key, tensor):
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
            "expert_representation": "split_unfused",
            "expert_banks_written": int(stats["expert_banks_written"]),
            "decoder_2d_quantized": int(stats["decoder_2d_quantized"]),
        },
    }
    del sd_new
    torch.cuda.empty_cache()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--format", choices=sorted(FORMAT_LAYOUTS), action="append", required=True)
    parser.add_argument("--out-mxfp8", type=Path, default=OUT_DEFAULTS["mxfp8"])
    parser.add_argument("--out-nvfp4", type=Path, default=OUT_DEFAULTS["nvfp4"])
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this converter")
    device = torch.device("cuda:0")
    outputs = {"mxfp8": args.out_mxfp8, "nvfp4": args.out_nvfp4}

    manifests = []
    for fmt in args.format:
        print(f"converting {fmt} -> {outputs[fmt]}")
        manifests.append(convert_one(args.src, outputs[fmt], fmt, device))

    args.manifest.write_text(json.dumps({"jobs": manifests}, indent=2, sort_keys=True) + "\n")
    print(args.manifest)


if __name__ == "__main__":
    main()
