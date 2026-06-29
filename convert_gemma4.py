#!/usr/bin/env python3
"""
Convert Google Gemma 4 checkpoints to ComfyUI safetensors.

The fp8_scaled precision intentionally reproduces Kijai's first public Gemma 4 FP8
conversion path: large 2D text weights are stored as float8_e4m3fn with scalar
weight_scale tensors and ComfyUI in-band comfy_quant metadata, while all other tensors
pass through unchanged. The defaults are chosen so a ComfyUI BF16 Gemma 4 E4B input can
reproduce Comfy-Org's gemma4_e4b_it_fp8_scaled.safetensors byte-for-byte.

Examples:
  # Reproduce the published E4B FP8-scaled artifact from the Comfy BF16 file.
  python convert_gemma4.py --src gemma4_e4b_it_bf16.safetensors \
      --job fp8_scaled:gemma4_e4b_it_fp8_scaled.safetensors:bf0b4fa2e41a25684dc9e9b256cd505564f02fed09be3da95ce024e653e2c52b

  # Convert a HuggingFace-layout safetensors file or snapshot directory to Comfy BF16.
  python convert_gemma4.py --src google-gemma4-snapshot --job bf16:gemma4_e4b_it_bf16.safetensors

  # One source load per job is not assumed; jobs are written one at a time to keep memory bounded.
  python convert_gemma4.py --src gemma4_e4b_it_bf16.safetensors \
      --job fp8_scaled:gemma4_e4b_it_fp8_scaled.safetensors
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

HF_TEXT_PREFIX = "model.language_model."
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


def tensor_row_chunks(tensor, rows):
    for start in range(0, tensor.shape[0], rows):
        end = min(start + rows, tensor.shape[0])
        yield start, end, tensor[start:end]


def fp8_scaled_scale(tensor, fp8_max, chunk_rows):
    amax = None
    for _, _, chunk in tensor_row_chunks(tensor, chunk_rows):
        chunk_amax = torch.max(torch.abs(chunk.float()))
        amax = chunk_amax if amax is None else torch.maximum(amax, chunk_amax)
    return amax / fp8_max


def quantize_fp8_scaled(key, tensor, fp8_max, chunk_rows):
    scale = fp8_scaled_scale(tensor, fp8_max, chunk_rows)
    q = torch.empty(tensor.shape, dtype=FP8_DTYPE)
    for start, end, chunk in tensor_row_chunks(tensor, chunk_rows):
        q[start:end] = (chunk.float() / scale).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(dtype=FP8_DTYPE)
    return {
        key: q,
        key.replace(".weight", ".weight_scale"): scale.cpu(),
        key.replace(".weight", ".comfy_quant"): quant_tensor(FP8_CONF),
    }


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


def convert(src, precision, text_only, tokenizer, fp8_max, chunk_rows):
    if precision == "fp8":
        precision = "fp8_scaled"
    if precision not in {"bf16", "fp16", "fp8_scaled"}:
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
                elif should_quantize_fp8_scaled(out_key, tensor):
                    out.update(quantize_fp8_scaled(out_key, tensor, fp8_max, chunk_rows))
                    stats["fp8_scaled"] += 1
                else:
                    out[out_key] = tensor
                    stats["passthrough"] += 1
                del tensor
        stats["source_files"] += 1
        print(f"processed {os.path.basename(path)}")

    add_tokenizer_if_needed(src, tokenizer, out, stats)
    return out, stats


def dump(src, text_only):
    stats = collections.Counter()
    examples = collections.defaultdict(list)
    for path in source_files(src):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if text_only and is_text_only_skip(key):
                    bucket = "skipped_text_only"
                    out_key = remap_key(key)
                    shape = tuple(f.get_slice(key).get_shape())
                else:
                    out_key = remap_key(key)
                    if text_only and is_text_only_skip(out_key):
                        bucket = "skipped_text_only"
                    else:
                        shape = tuple(f.get_slice(key).get_shape())
                        fake = ShapeOnlyTensor(shape)
                        bucket = "fp8_scaled" if should_quantize_fp8_scaled(out_key, fake) else "passthrough"
                stats[bucket] += 1
                if len(examples[bucket]) < 8:
                    examples[bucket].append((key, out_key, shape))
    print("stats " + json.dumps(dict(sorted(stats.items())), sort_keys=True))
    for bucket in sorted(examples):
        print(bucket)
        for key, out_key, shape in examples[bucket]:
            print(f"  {key} -> {out_key} {shape}")


def write_job(src, job, text_only, tokenizer, fp8_max, chunk_rows):
    precision, out_path, expected = parse_job(job)
    sd, stats = convert(src, precision, text_only, tokenizer, fp8_max, chunk_rows)
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
                    help="repeatable; PRECISION is bf16, fp16, fp8_scaled, or fp8 alias")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json path if source lacks tokenizer_json")
    ap.add_argument("--text-only", action="store_true", help="drop vision/audio tower and projector tensors")
    ap.add_argument("--fp8-max", type=float, default=KIJAI_FP8_MAX,
                    help="FP8 scale divisor; default 416 matches the published Gemma 4 E4B FP8 artifact")
    ap.add_argument("--chunk-rows", type=int, default=4096,
                    help="rows per FP8 conversion chunk; affects peak RAM but not the quantization policy")
    ap.add_argument("--dump", action="store_true", help="print tensor routing without writing outputs")
    args = ap.parse_args()

    if args.dump:
        dump(args.src, args.text_only)
    if not args.job:
        if args.dump:
            return
        raise SystemExit("at least one --job is required unless --dump is set")
    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be positive")
    for job in args.job:
        write_job(args.src, job, args.text_only, args.tokenizer, args.fp8_max, args.chunk_rows)


if __name__ == "__main__":
    main()
