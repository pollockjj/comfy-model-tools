#!/usr/bin/env python3
"""
Convert Google Gemma 4 checkpoints to ComfyUI safetensors.

This V2 shell follows the SeedVR2 converter shape: explicit jobs, explicit routing
stats, and optional SHA verification on every written artifact. The first precision
implemented here is fp8_scaled, intentionally matching Kijai's Gemma 4 FP8 V1 behavior
byte-for-byte:

  - quantize large 2D text weights under model.*
  - keep norms, small 2D weights, embeddings below the policy threshold, vision/audio,
    projectors, tokenizer_json, and scalar metadata unchanged
  - use float8_e4m3fn storage, scalar weight_scale, and ComfyUI comfy_quant metadata
  - use scale divisor 416.0 and default json.dumps metadata bytes

The int8_convrot precision emits ComfyUI-native TensorWiseINT8Layout weights
for eligible text linears:

  - quantize non-embedding 2D text weights under model.*
  - require the input dimension to be divisible by the ConvRot group size
  - keep embeddings, norms, vision/audio, projectors, tokenizer_json, and scalar
    metadata unchanged
  - use int8_tensorwise storage, per-channel scales, and ConvRot group size 256

Examples:
  python gemma4_convert_v2.py --src gemma4_e4b_it_bf16.safetensors \
      --job fp8_scaled:gemma4_e4b_it_fp8_scaled.safetensors:bf0b4fa2e41a25684dc9e9b256cd505564f02fed09be3da95ce024e653e2c52b

  python gemma4_convert_v2.py --src gemma4_e4b_it_bf16.safetensors \
      --job int8_convrot:gemma4_e4b_it_int8_convrot.safetensors

  python gemma4_convert_v2.py --src google-gemma4-snapshot \
      --job bf16:gemma4_e4b_it_bf16.safetensors
"""
import argparse
import collections
import glob
import hashlib
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_FORMAT = "float8_e4m3fn"
FP8_DTYPE = torch.float8_e4m3fn
FP8_INFO = torch.finfo(FP8_DTYPE)
KIJAI_FP8_MAX = 416.0
FP8_CONF = {"format": FP8_FORMAT, "full_precision_matrix_mult": False}
INT8_FORMAT = "int8_tensorwise"
INT8_LAYOUT = "TensorWiseINT8Layout"
INT8_CONVROT_GROUPSIZE = 256
INT8_CONF = {
    "format": INT8_FORMAT,
    "convrot": True,
    "convrot_groupsize": INT8_CONVROT_GROUPSIZE,
}

HF_PREFIX_REMAPS = (
    ("model.language_model.", "model."),
    ("model.vision_tower.", "vision_model."),
    ("model.audio_tower.", "audio_model."),
    ("model.embed_vision.", "multi_modal_projector."),
    ("model.embed_audio.", "audio_projector."),
)
TEXT_ONLY_SKIP_PREFIXES = (
    "model.vision_tower.",
    "model.audio_tower.",
    "model.embed_vision.",
    "model.embed_audio.",
    "vision_model.",
    "audio_model.",
    "multi_modal_projector.",
    "audio_projector.",
)


class ShapeOnlyTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def dim(self):
        return len(self.shape)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_bytes(sd):
    return sum(t.numel() * t.element_size() for t in sd.values())


def source_files(src):
    if os.path.isfile(src):
        return [src]
    if not os.path.isdir(src):
        raise SystemExit(f"source does not exist: {src}")
    files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not files:
        raise SystemExit(f"no .safetensors files found in {src}")
    return files


def parse_job(job):
    precision, sep, remainder = job.partition(":")
    if not sep or not remainder:
        raise SystemExit(f"invalid --job {job!r}; expected PRECISION:OUT[:SHA256]")
    out_path = remainder
    expected = None
    head, sha_sep, tail = remainder.rpartition(":")
    if sha_sep and len(tail) == 64 and all(c in "0123456789abcdefABCDEF" for c in tail):
        out_path = head
        expected = tail.lower()
    return precision, out_path, expected


def remap_key(key):
    for old, new in HF_PREFIX_REMAPS:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def is_text_only_skip(key):
    return key.startswith(TEXT_ONLY_SKIP_PREFIXES)


def quant_tensor(conf):
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def cast_floating(tensor, dtype):
    if torch.is_floating_point(tensor):
        return tensor.to(dtype)
    return tensor


def should_quantize_fp8_scaled(key, tensor):
    return (
        key.startswith("model.")
        and key.endswith(".weight")
        and tensor.dim() == 2
        and "norm" not in key
        and max(tensor.shape) >= 4096
    )


def should_quantize_int8_convrot(key, tensor):
    return (
        key.startswith("model.")
        and key.endswith(".weight")
        and tensor.dim() == 2
        and "norm" not in key
        and "embed_tokens" not in key
        and tensor.shape[1] % INT8_CONVROT_GROUPSIZE == 0
    )


def quantize_fp8_scaled(key, tensor, fp8_max):
    w = tensor.float()
    scale = torch.max(torch.abs(w)) / fp8_max
    q = (w / scale).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(dtype=FP8_DTYPE)
    return {
        key: q,
        key.replace(".weight", ".weight_scale"): scale.cpu(),
        key.replace(".weight", ".comfy_quant"): quant_tensor(FP8_CONF),
    }


def quantize_int8_convrot(key, tensor):
    try:
        from comfy_kitchen.tensor import QuantizedTensor
    except ImportError as e:
        raise SystemExit("int8_convrot precision requires comfy-kitchen") from e

    qt = QuantizedTensor.from_float(
        tensor.contiguous(),
        INT8_LAYOUT,
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=INT8_CONVROT_GROUPSIZE,
    )
    tensors = qt.state_dict(key)
    tensors[key.replace(".weight", ".comfy_quant")] = quant_tensor(INT8_CONF)
    return tensors


def tokenizer_candidates(src, tokenizer):
    if tokenizer:
        return [tokenizer]
    if os.path.isdir(src):
        return [os.path.join(src, "tokenizer.json")]
    return [os.path.join(os.path.dirname(src), "tokenizer.json")]


def add_tokenizer_if_needed(src, tokenizer, out, stats):
    if "tokenizer_json" in out:
        stats["tokenizer_json_passthrough"] += 1
        return
    for candidate in tokenizer_candidates(src, tokenizer):
        if os.path.exists(candidate):
            with open(candidate, "rb") as f:
                out["tokenizer_json"] = torch.tensor(list(f.read()), dtype=torch.uint8)
            stats["tokenizer_json_added"] += 1
            return
    raise SystemExit("missing tokenizer_json tensor and no tokenizer.json found; pass --tokenizer")


def convert(src, precision, text_only, tokenizer, fp8_max):
    if precision == "fp8":
        precision = "fp8_scaled"
    if precision == "int8":
        precision = "int8_convrot"
    if precision not in {"bf16", "fp16", "fp8_scaled", "int8_convrot"}:
        raise SystemExit(f"unknown precision: {precision}")

    out = {}
    stats = collections.Counter()
    for path in source_files(src):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if text_only and is_text_only_skip(key):
                    stats["skipped_text_only"] += 1
                    continue
                out_key = remap_key(key)
                if text_only and is_text_only_skip(out_key):
                    stats["skipped_text_only"] += 1
                    continue
                if out_key in out:
                    raise SystemExit(f"duplicate output key after remap: {out_key}")

                tensor = f.get_tensor(key)
                if precision == "bf16":
                    out[out_key] = cast_floating(tensor, torch.bfloat16)
                    stats["bf16"] += 1
                elif precision == "fp16":
                    out[out_key] = cast_floating(tensor, torch.float16)
                    stats["fp16"] += 1
                elif precision == "fp8_scaled" and should_quantize_fp8_scaled(out_key, tensor):
                    out.update(quantize_fp8_scaled(out_key, tensor, fp8_max))
                    stats["fp8_scaled"] += 1
                elif precision == "int8_convrot" and should_quantize_int8_convrot(out_key, tensor):
                    out.update(quantize_int8_convrot(out_key, tensor))
                    stats["int8_convrot"] += 1
                else:
                    out[out_key] = tensor
                    if (
                        precision == "int8_convrot"
                        and out_key.startswith("model.")
                        and out_key.endswith(".weight")
                        and tensor.dim() == 2
                        and "embed_tokens" in out_key
                    ):
                        stats["int8_embedding_passthrough"] += 1
                    stats["passthrough"] += 1
                del tensor
        stats["source_files"] += 1
        print(f"processed {os.path.basename(path)}")

    add_tokenizer_if_needed(src, tokenizer, out, stats)
    return out, stats


def dump_bucket(precision, out_key, tensor):
    if precision == "fp8":
        precision = "fp8_scaled"
    if precision == "int8":
        precision = "int8_convrot"
    if precision == "fp8_scaled":
        return "fp8_scaled" if should_quantize_fp8_scaled(out_key, tensor) else "passthrough"
    if precision == "int8_convrot":
        return "int8_convrot" if should_quantize_int8_convrot(out_key, tensor) else "passthrough"
    raise SystemExit(f"unknown dump precision: {precision}")


def dump(src, text_only, precision):
    stats = collections.Counter()
    examples = collections.defaultdict(list)
    for path in source_files(src):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                shape = tuple(f.get_slice(key).get_shape())
                out_key = remap_key(key)
                if text_only and (is_text_only_skip(key) or is_text_only_skip(out_key)):
                    bucket = "skipped_text_only"
                else:
                    fake = ShapeOnlyTensor(shape)
                    bucket = dump_bucket(precision, out_key, fake)
                stats[bucket] += 1
                if len(examples[bucket]) < 8:
                    examples[bucket].append((key, out_key, shape))
    print("stats " + json.dumps(dict(sorted(stats.items())), sort_keys=True))
    for bucket in sorted(examples):
        print(bucket)
        for key, out_key, shape in examples[bucket]:
            print(f"  {key} -> {out_key} {shape}")


def write_job(src, job, text_only, tokenizer, fp8_max):
    precision, out_path, expected = parse_job(job)
    sd, stats = convert(src, precision, text_only, tokenizer, fp8_max)
    save_file(sd, out_path)
    digest = sha256(out_path)
    size_gb = tensor_bytes(sd) / 1024**3
    print(f"{precision:10s} {digest} {size_gb:.2f} GB {out_path}")
    print("stats " + json.dumps(dict(sorted(stats.items())), sort_keys=True))
    if expected is not None and digest != expected:
        raise SystemExit(f"SHA256 mismatch for {out_path}: expected {expected}, got {digest}")


def main():
    ap = argparse.ArgumentParser(description="Convert Google Gemma 4 checkpoints to ComfyUI safetensors.")
    ap.add_argument("--src", required=True, help="source safetensors file or directory")
    ap.add_argument("--job", action="append", metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; PRECISION is bf16, fp16, fp8_scaled, fp8 alias, int8_convrot, or int8 alias")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json path if source lacks tokenizer_json")
    ap.add_argument("--text-only", action="store_true", help="drop vision/audio tower and projector tensors")
    ap.add_argument("--fp8-max", type=float, default=KIJAI_FP8_MAX,
                    help="FP8 scale divisor; default 416 matches the published Gemma 4 E4B FP8 artifact")
    ap.add_argument("--dump-precision", default="fp8_scaled",
                    help="precision policy to report with --dump; default fp8_scaled")
    ap.add_argument("--dump", action="store_true", help="print tensor routing without writing outputs")
    args = ap.parse_args()

    if args.dump:
        dump(args.src, args.text_only, args.dump_precision)
    if not args.job:
        if args.dump:
            return
        raise SystemExit("at least one --job is required unless --dump is set")
    for job in args.job:
        write_job(args.src, job, args.text_only, args.tokenizer, args.fp8_max)


if __name__ == "__main__":
    main()
