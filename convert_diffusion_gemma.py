#!/usr/bin/env python3
"""
Convert DiffusionGemma-26B-A4B checkpoints to ComfyUI text-encoder safetensors.

The converter follows the local SeedVR2 conversion pattern: one source inventory,
repeatable output jobs, explicit quantization policy, SHA256 verification, and a
manifest for every written artifact.

Supported source forms:
  - Hugging Face snapshot directory with model-00001-of-00011.safetensors shards.
  - Existing ComfyUI BF16 text-encoder safetensors.

Supported jobs:
  - bf16     ComfyUI BF16 text encoder.
  - fp8_v2   FP8 E4M3 V2 candidate.

Examples:
  python convert_diffusion_gemma.py \
    --src /home/johnj/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/0f28bc42f588fbd8f71e08102b1c3960298a1358 \
    --job bf16:/home/johnj/dev_master/ComfyUI/models/text_encoders/diffusiongemma_comfy_bf16.safetensors \
    --job fp8_v2:/home/johnj/dev_master/ComfyUI/models/text_encoders/diffusiongemma_comfy_fp8v2.safetensors \
    --manifest-dir /home/johnj/dev_master/mydevelopment/github_issues/319/scratch

Legacy shape is still accepted:
  python convert_diffusion_gemma.py <hf_snapshot_or_bf16.safetensors> <out.safetensors> [--bf16]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


EXPECTED_SHARDS = 11
EXPECTED_LAYERS = 30
EXPECTED_EXPERTS = 128
FP8_LAYOUT = "TensorCoreFP8Layout"
FP8_FORMAT = "float8_e4m3fn"
FP8_DTYPE = torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8_DTYPE)
FP8_MAX_VALUE = FP8_INFO.max

EXPERT_HF_SUFFIXES = (".experts.gate_up_proj", ".experts.down_proj")
EXPERT_COMFY_SUFFIXES = (".experts.gate_up_proj.weight", ".experts.down_proj.weight")
ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_ATTN_COUNTS = {"q_proj": 30, "k_proj": 30, "v_proj": 25, "o_proj": 30}
EXPECTED_MLP_COUNTS = {"gate_proj": 30, "up_proj": 30, "down_proj": 30}


def sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_nbytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors.values())


def quant_json(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf, sort_keys=True).encode("utf-8")), dtype=torch.uint8)


def parse_job(spec: str) -> tuple[str, str, str | None]:
    precision, sep, remainder = spec.partition(":")
    if not sep or not remainder:
        raise SystemExit(f"invalid --job {spec!r}; expected PRECISION:OUT[:SHA256]")
    out, expected = remainder, None
    head, sha_sep, tail = remainder.rpartition(":")
    if sha_sep and len(tail) == 64 and all(c in "0123456789abcdefABCDEF" for c in tail):
        out, expected = head, tail.lower()
    precision = precision.lower()
    if precision == "fp8v2":
        precision = "fp8_v2"
    if precision not in {"bf16", "fp8_v2"}:
        raise SystemExit(f"unknown precision {precision!r}; supported: bf16, fp8_v2")
    return precision, out, expected


def source_revision(path: Path) -> str | None:
    parts = path.resolve().parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(path.name) >= 12 and re.fullmatch(r"[0-9a-fA-F]{12,}", path.name):
        return path.name
    return None


def load_config(snapshot: Path) -> dict:
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"missing config.json in HF snapshot: {snapshot}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    text_config = config.get("text_config", {})
    layers = text_config.get("num_hidden_layers")
    experts = text_config.get("num_experts")
    if layers != EXPECTED_LAYERS or experts != EXPECTED_EXPERTS:
        raise SystemExit(
            f"unexpected DG config: num_hidden_layers={layers}, num_experts={experts}; "
            f"expected {EXPECTED_LAYERS}/{EXPECTED_EXPERTS}"
        )
    return config


def normalize_hf_key(key: str) -> str | None:
    if key.startswith("lm_head."):
        return None
    if key.endswith(EXPERT_HF_SUFFIXES):
        return f"{key}.weight"
    return key


def load_hf_snapshot(snapshot: Path) -> tuple[dict[str, torch.Tensor], dict]:
    if not snapshot.is_dir():
        raise SystemExit(f"HF source is not a directory: {snapshot}")
    config = load_config(snapshot)
    shards = sorted(snapshot.glob("model-*-of-*.safetensors"))
    if len(shards) != EXPECTED_SHARDS:
        raise SystemExit(f"expected {EXPECTED_SHARDS} HF shards, found {len(shards)} in {snapshot}")
    tokenizer_path = snapshot / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise SystemExit(f"missing tokenizer.json in HF snapshot: {snapshot}")

    out: dict[str, torch.Tensor] = {}
    source_keys = 0
    skipped_lm_head = 0
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                source_keys += 1
                out_key = normalize_hf_key(key)
                if out_key is None:
                    skipped_lm_head += 1
                    continue
                if out_key in out:
                    raise SystemExit(f"duplicate output key after HF mapping: {out_key}")
                out[out_key] = f.get_tensor(key)
        print(f"processed {shard.name}")

    with open(tokenizer_path, "rb") as f:
        out["tokenizer_json"] = torch.tensor(list(f.read()), dtype=torch.uint8)

    meta = {
        "source_type": "hf_snapshot",
        "source_path": str(snapshot),
        "source_revision": source_revision(snapshot),
        "hf_source_keys": source_keys,
        "hf_skipped_lm_head": skipped_lm_head,
        "hf_shards": len(shards),
        "config_num_layers": config["text_config"]["num_hidden_layers"],
        "config_num_experts": config["text_config"]["num_experts"],
    }
    validate_comfy_inventory(out)
    return out, meta


def load_comfy_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    if not path.is_file():
        raise SystemExit(f"Comfy source is not a file: {path}")
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    meta = {
        "source_type": "comfy_safetensors",
        "source_path": str(path),
        "source_revision": None,
    }
    validate_comfy_inventory(out)
    return out, meta


def load_source(src: str) -> tuple[dict[str, torch.Tensor], dict]:
    path = Path(src)
    if path.is_dir():
        return load_hf_snapshot(path)
    return load_comfy_safetensors(path)


def decoder_layer_id(key: str) -> int | None:
    prefix = "model.decoder.layers."
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix):]
    layer, sep, _ = rest.partition(".")
    if not sep or not layer.isdigit():
        return None
    return int(layer)


def attention_projection(key: str) -> str | None:
    if not key.startswith("model.decoder.layers.") or not key.endswith(".weight"):
        return None
    for name in ATTN_PROJECTIONS:
        if f".self_attn.{name}.weight" in key:
            return name
    return None


def mlp_projection(key: str) -> str | None:
    if not key.startswith("model.decoder.layers.") or not key.endswith(".weight"):
        return None
    for name in MLP_PROJECTIONS:
        if f".mlp.{name}.weight" in key:
            return name
    return None


def is_expert_bank(key: str) -> bool:
    return key.endswith(EXPERT_COMFY_SUFFIXES)


def kept_family(key: str) -> str:
    if key == "tokenizer_json":
        return "tokenizer"
    if "embed" in key:
        return "embeddings"
    if key.startswith("model.encoder."):
        return "vision"
    if ".router." in key:
        return "router"
    if "self_condition" in key or "self_cond" in key:
        return "self_conditioning"
    if "norm" in key or key.endswith(".scale") or key.endswith(".layer_scalar"):
        return "norms_scalars"
    if key.startswith("model.decoder."):
        return "decoder_other"
    return "other"


def validate_comfy_inventory(sd: dict[str, torch.Tensor]) -> None:
    layers = {layer for key in sd for layer in [decoder_layer_id(key)] if layer is not None}
    expert_banks = sum(1 for key in sd if is_expert_bank(key))
    if len(layers) != EXPECTED_LAYERS:
        raise SystemExit(f"expected {EXPECTED_LAYERS} decoder layers, found {len(layers)}")
    if expert_banks != EXPECTED_LAYERS * 2:
        raise SystemExit(f"expected {EXPECTED_LAYERS * 2} expert banks, found {expert_banks}")
    if "tokenizer_json" not in sd:
        raise SystemExit("missing tokenizer_json payload")


def quantize_2d_weight(key: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    if tensor.dim() != 2:
        raise SystemExit(f"FP8 2D quantization expected 2D tensor for {key}, got {tuple(tensor.shape)}")
    try:
        from comfy_kitchen.tensor import QuantizedTensor
    except ImportError as e:
        raise SystemExit("fp8_v2 requires comfy-kitchen") from e

    qt = QuantizedTensor.from_float(tensor.contiguous(), FP8_LAYOUT)
    tensors = qt.state_dict(key)
    tensors[key.replace(".weight", ".comfy_quant")] = quant_json({"format": FP8_FORMAT})
    return tensors


def quantize_expert_bank(key: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    if tensor.dim() != 3:
        raise SystemExit(f"FP8 expert-bank quantization expected 3D tensor for {key}, got {tuple(tensor.shape)}")
    if tensor.shape[0] != EXPECTED_EXPERTS:
        raise SystemExit(f"expected {EXPECTED_EXPERTS} experts for {key}, got {tensor.shape[0]}")

    qdata = torch.empty(tensor.shape, dtype=FP8_DTYPE, device="cpu")
    scales = torch.empty((tensor.shape[0],), dtype=torch.float32, device="cpu")
    for expert_idx in range(tensor.shape[0]):
        expert = tensor[expert_idx].float()
        scale = torch.amax(expert.abs()) / FP8_MAX_VALUE
        if scale.item() == 0:
            raise SystemExit(f"zero FP8 scale for {key} expert {expert_idx}")
        scales[expert_idx] = scale
        qdata[expert_idx] = (expert / scale).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(dtype=FP8_DTYPE)

    return {
        key: qdata,
        key.replace(".weight", ".weight_scale"): scales,
        key.replace(".weight", ".comfy_quant"): quant_json(
            {
                "format": FP8_FORMAT,
                "num_experts": tensor.shape[0],
                "scale_granularity": "per_expert",
            }
        ),
    }


def cast_bf16(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    out: dict[str, torch.Tensor] = {}
    dtype_counts = collections.Counter()
    for key, tensor in sd.items():
        if key == "tokenizer_json":
            out[key] = tensor
        elif torch.is_floating_point(tensor):
            out[key] = tensor.to(torch.bfloat16)
        else:
            out[key] = tensor
        dtype_counts[str(out[key].dtype)] += 1
    return out, {"dtype_counts": dict(dtype_counts)}


def cast_fp8_v2(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    out: dict[str, torch.Tensor] = {}
    coverage = {
        "expert_banks": 0,
        "attention": collections.Counter(),
        "dense_mlp": collections.Counter(),
        "metadata_tensors": 0,
    }
    kept_bf16 = collections.Counter()

    for key, tensor in sd.items():
        attn_name = attention_projection(key)
        mlp_name = mlp_projection(key)
        if is_expert_bank(key):
            print(f"quantizing expert bank {key}")
            quantized = quantize_expert_bank(key, tensor)
            out.update(quantized)
            coverage["expert_banks"] += 1
            coverage["metadata_tensors"] += 1
        elif attn_name is not None:
            print(f"quantizing attention {attn_name} {key}")
            quantized = quantize_2d_weight(key, tensor)
            out.update(quantized)
            coverage["attention"][attn_name] += 1
            coverage["metadata_tensors"] += 1
        elif mlp_name is not None:
            print(f"quantizing dense MLP {mlp_name} {key}")
            quantized = quantize_2d_weight(key, tensor)
            out.update(quantized)
            coverage["dense_mlp"][mlp_name] += 1
            coverage["metadata_tensors"] += 1
        else:
            if torch.is_floating_point(tensor):
                out[key] = tensor.to(torch.bfloat16)
            else:
                out[key] = tensor
            kept_bf16[kept_family(key)] += 1

    attention_counts = dict(coverage["attention"])
    dense_counts = dict(coverage["dense_mlp"])
    if coverage["expert_banks"] != EXPECTED_LAYERS * 2:
        raise SystemExit(f"fp8_v2 expert coverage mismatch: {coverage['expert_banks']}")
    if attention_counts != EXPECTED_ATTN_COUNTS:
        raise SystemExit(f"fp8_v2 attention coverage mismatch: {attention_counts}")
    if dense_counts != EXPECTED_MLP_COUNTS:
        raise SystemExit(f"fp8_v2 dense MLP coverage mismatch: {dense_counts}")

    manifest_bits = {
        "format": FP8_FORMAT,
        "representation": "bank3d",
        "compute_claim": (
            "mixed_storage_first: 2D Linear weights can use native FP8 on SM8.9+ when "
            "runtime full_precision_mm is disabled; grouped 3D expert banks remain "
            "storage-first until native grouped-bank execution is proven"
        ),
        "coverage": {
            "expert_banks": coverage["expert_banks"],
            "expert_bank_scale": "per_expert",
            "attention": attention_counts,
            "dense_mlp": dense_counts,
            "metadata_tensors": coverage["metadata_tensors"],
        },
        "kept_bf16": dict(kept_bf16),
        "skips": {
            "policy": [
                "embeddings",
                "router",
                "self_conditioning",
                "vision",
                "norms_scalars",
                "tokenizer",
            ],
            "shape": {},
            "unsupported_layout": {},
        },
        "failures": [],
    }
    return out, manifest_bits


def manifest_path_for(out_path: Path, manifest_dir: str | None) -> Path:
    if manifest_dir:
        return Path(manifest_dir) / f"{out_path.name}.manifest.json"
    return out_path.with_suffix(out_path.suffix + ".manifest.json")


def write_manifest(
    out_path: Path,
    manifest_dir: str | None,
    precision: str,
    source_meta: dict,
    source_tensor_count: int,
    source_bytes: int,
    output_tensors: dict[str, torch.Tensor],
    digest: str,
    extra: dict,
) -> Path:
    manifest_path = manifest_path_for(out_path, manifest_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dtype_counts = collections.Counter(str(t.dtype) for t in output_tensors.values())
    manifest = {
        "source_path": source_meta.get("source_path"),
        "source_type": source_meta.get("source_type"),
        "source_revision": source_meta.get("source_revision"),
        "source_tensor_count": source_tensor_count,
        "source_bytes": source_bytes,
        "output_path": str(out_path),
        "precision": precision,
        "tensor_count": len(output_tensors),
        "bytes": tensor_nbytes(output_tensors),
        "sha256": digest,
        "dtype_counts": dict(dtype_counts),
    }
    manifest.update({k: v for k, v in source_meta.items() if k not in manifest})
    manifest.update(extra)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path


def dump_inventory(sd: dict[str, torch.Tensor]) -> None:
    dtypes = collections.Counter(str(t.dtype) for t in sd.values())
    attn = collections.Counter(attention_projection(k) for k in sd if attention_projection(k) is not None)
    mlp = collections.Counter(mlp_projection(k) for k in sd if mlp_projection(k) is not None)
    print(f"tensors={len(sd)} bytes={tensor_nbytes(sd)} dtypes={dict(dtypes)}")
    print(f"expert_banks={sum(1 for k in sd if is_expert_bank(k))}")
    print(f"attention={dict(attn)}")
    print(f"dense_mlp={dict(mlp)}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert DiffusionGemma checkpoints to ComfyUI safetensors.")
    ap.add_argument("legacy", nargs="*", help=argparse.SUPPRESS)
    ap.add_argument("--src", help="HF snapshot directory or Comfy BF16 safetensors")
    ap.add_argument("--job", action="append", metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; supported precisions: bf16, fp8_v2")
    ap.add_argument("--manifest-dir", default=None, help="directory for output manifests")
    ap.add_argument("--dump", action="store_true", help="print source inventory")
    ap.add_argument("--bf16", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.legacy:
        if args.src or args.job:
            raise SystemExit("legacy positional mode cannot be combined with --src/--job")
        if len(args.legacy) != 2:
            raise SystemExit("legacy mode expects: <hf_snapshot_or_bf16.safetensors> <out.safetensors> [--bf16]")
        args.src = args.legacy[0]
        precision = "bf16" if args.bf16 else "fp8_v2"
        args.job = [f"{precision}:{args.legacy[1]}"]
    if not args.src:
        raise SystemExit("--src is required")
    if not args.job and not args.dump:
        raise SystemExit("at least one --job is required unless --dump is used")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    sd, source_meta = load_source(args.src)
    source_tensor_count = len(sd)
    source_bytes = tensor_nbytes(sd)

    if args.dump:
        dump_inventory(sd)

    mismatched = []
    for job in args.job or []:
        precision, out, expected = parse_job(job)
        out_path = Path(out)
        if precision == "bf16":
            tensors, extra = cast_bf16(sd)
            extra.setdefault("format", "bfloat16")
            extra.setdefault("representation", "bank3d")
            extra.setdefault("compute_claim", "bf16_oracle")
        elif precision == "fp8_v2":
            tensors, extra = cast_fp8_v2(sd)
        else:
            raise AssertionError(precision)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out_path))
        digest = sha256(out_path)
        manifest_path = write_manifest(
            out_path,
            args.manifest_dir,
            precision,
            source_meta,
            source_tensor_count,
            source_bytes,
            tensors,
            digest,
            extra,
        )
        verdict = "" if expected is None else ("  OK" if digest == expected else "  MISMATCH")
        print(f"{precision:8s} {digest}  {out_path}{verdict}")
        print(f"manifest {manifest_path}")
        if expected is not None and digest != expected:
            mismatched.append(str(out_path))

    if mismatched:
        raise SystemExit(f"SHA256 mismatch: {', '.join(mismatched)}")


if __name__ == "__main__":
    main()
