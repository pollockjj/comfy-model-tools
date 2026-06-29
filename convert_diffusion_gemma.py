#!/usr/bin/env python3
"""Convert DiffusionGemma HuggingFace shards to ComfyUI safetensors."""
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
DEFAULT_FP8_MAX = float(FP8_INFO.max)
GATE_UP_EXPERT_SUFFIX = ".experts.gate_up_proj"
DOWN_EXPERT_SUFFIX = ".experts.down_proj"
EXPERT_BANK_SUFFIXES = (GATE_UP_EXPERT_SUFFIX, DOWN_EXPERT_SUFFIX)
MLP_WEIGHT_SUFFIXES = (".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")
ATTN_WEIGHT_SUFFIXES = (".self_attn.q_proj.weight", ".self_attn.o_proj.weight")
EMBED_TOKENS_KEY = "model.decoder.embed_tokens.weight"


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


def quant_tensor(conf):
    data = json.dumps(conf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(data), dtype=torch.uint8)


def fp8_conf(**extra):
    conf = {"format": FP8_FORMAT, "full_precision_matrix_mult": False}
    conf.update(extra)
    return quant_tensor(conf)


def fail_nonfinite(key, tensor):
    if not torch.isfinite(tensor).all():
        raise SystemExit(f"{key}: non-finite values cannot be quantized")


def scalar_scale(key, tensor, max_value):
    amax = torch.amax(torch.abs(tensor)).to(torch.float32)
    fail_nonfinite(key, amax)
    if amax.item() == 0.0:
        return torch.ones((), dtype=torch.float32, device=tensor.device)
    return amax / max_value


def expert_scales(key, tensor, max_value):
    amax = torch.amax(torch.abs(tensor), dim=(1, 2)).to(torch.float32)
    fail_nonfinite(key, amax)
    return torch.where(amax == 0, torch.ones_like(amax), amax / max_value)


def floating_passthrough(tensor):
    if torch.is_floating_point(tensor) and tensor.dtype != torch.bfloat16:
        return tensor.to(torch.bfloat16)
    return tensor


def quantize_2d(key, tensor, max_value):
    w = tensor.float()
    scale = scalar_scale(key, w, max_value)
    q = (w / scale).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(dtype=FP8_DTYPE)
    return {
        key: q,
        key.replace(".weight", ".weight_scale"): scale.cpu(),
        key.replace(".weight", ".comfy_quant"): fp8_conf(),
    }


def quantize_bank(key, tensor, max_value):
    w = tensor.float()
    scale = expert_scales(key, w, max_value)
    q = (w / scale[:, None, None]).clamp(min=FP8_INFO.min, max=FP8_INFO.max).to(dtype=FP8_DTYPE)
    return {
        f"{key}.weight": q,
        f"{key}.weight_scale": scale.cpu(),
        f"{key}.comfy_quant": fp8_conf(num_experts=w.shape[0]),
    }


def quantize_split_gate_up_bank(key, tensor, max_value):
    if tensor.dim() != 3:
        raise SystemExit(f"{key}: expected 3D fused expert bank, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] % 2:
        raise SystemExit(f"{key}: cannot split odd gate_up output dimension {tensor.shape[1]}")

    half = tensor.shape[1] // 2
    gate, up = tensor.split(half, dim=1)
    prefix = key[: -len(GATE_UP_EXPERT_SUFFIX)]
    out = {}
    out.update(quantize_bank(f"{prefix}.experts.gate_proj", gate, max_value))
    out.update(quantize_bank(f"{prefix}.experts.up_proj", up, max_value))
    return out


def is_decoder_layer_weight(key, tensor):
    return (
        key.startswith("model.decoder.layers.")
        and key.endswith(".weight")
        and tensor.dim() == 2
        and "norm" not in key
    )


def should_quantize_fp8(key, tensor, policy, expert_layout):
    if key.endswith(GATE_UP_EXPERT_SUFFIX):
        return "expert_split_gate_up_bank" if expert_layout == "split" else "expert_bank"
    if key.endswith(DOWN_EXPERT_SUFFIX):
        return "expert_split_down_bank" if expert_layout == "split" else "expert_bank"
    if key.endswith(EXPERT_BANK_SUFFIXES):
        return "expert_bank"
    if key == EMBED_TOKENS_KEY:
        return "embedding" if policy == "balanced" else None
    if not is_decoder_layer_weight(key, tensor):
        return None
    if key.endswith(ATTN_WEIGHT_SUFFIXES):
        return "attention"
    if policy == "balanced" and key.endswith(MLP_WEIGHT_SUFFIXES):
        return "mlp"
    return None


def discover_shards(src):
    shards = sorted(glob.glob(os.path.join(src, "model-*-of-*.safetensors")))
    if not shards:
        raise SystemExit(f"no model-*-of-*.safetensors shards found in {src}")
    tokenizer = os.path.join(src, "tokenizer.json")
    if not os.path.exists(tokenizer):
        raise SystemExit(f"missing tokenizer.json in {src}")
    return shards, tokenizer


def convert(src, precision, fp8_policy, fp8_max, fp8_expert_layout):
    if precision not in {"bf16", "fp8"}:
        raise SystemExit(f"unknown precision: {precision}")

    shards, tokenizer = discover_shards(src)
    out = {}
    stats = collections.Counter()
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key.startswith("lm_head."):
                    stats["skipped_lm_head"] += 1
                    continue

                quant_kind = should_quantize_fp8(key, tensor, fp8_policy, fp8_expert_layout) if precision == "fp8" else None
                if quant_kind == "expert_bank":
                    out.update(quantize_bank(key, tensor, fp8_max))
                    stats["fp8_expert_banks"] += 1
                elif quant_kind == "expert_split_gate_up_bank":
                    out.update(quantize_split_gate_up_bank(key, tensor, fp8_max))
                    stats["fp8_expert_split_gate_up_banks"] += 1
                elif quant_kind == "expert_split_down_bank":
                    out.update(quantize_bank(key, tensor, fp8_max))
                    stats["fp8_expert_split_down_banks"] += 1
                elif quant_kind is not None:
                    out.update(quantize_2d(key, tensor, fp8_max))
                    stats[f"fp8_{quant_kind}"] += 1
                elif key.endswith(EXPERT_BANK_SUFFIXES):
                    out[f"{key}.weight"] = floating_passthrough(tensor)
                    stats["bf16_expert_banks"] += 1
                else:
                    out[key] = floating_passthrough(tensor)
                    stats["passthrough"] += 1
                del tensor
        print(f"processed {os.path.basename(shard)}")

    with open(tokenizer, "rb") as f:
        out["tokenizer_json"] = torch.tensor(list(f.read()), dtype=torch.uint8)
    stats["tokenizer_json"] = 1
    return out, stats


def tensor_bytes(sd):
    return sum(t.numel() * t.element_size() for t in sd.values())


def write_job(src, job, fp8_policy, fp8_max, fp8_expert_layout):
    precision, sep, remainder = job.partition(":")
    if not sep or not remainder:
        raise SystemExit(f"invalid --job {job!r}; expected PRECISION:OUT[:SHA256]")

    out_path = remainder
    expected = None
    head, sha_sep, tail = remainder.rpartition(":")
    if sha_sep and len(tail) == 64 and all(c in "0123456789abcdefABCDEF" for c in tail):
        out_path = head
        expected = tail.lower()

    sd, stats = convert(src, precision, fp8_policy, fp8_max, fp8_expert_layout)
    save_file(sd, out_path)
    digest = sha256(out_path)
    size_gb = tensor_bytes(sd) / 1024**3
    print(f"{precision:4s} {digest} {size_gb:.2f} GB {out_path}")
    print("stats " + json.dumps(dict(sorted(stats.items())), sort_keys=True))
    if expected is not None and digest != expected:
        raise SystemExit(f"SHA256 mismatch for {out_path}: expected {expected}, got {digest}")


def dump(src, fp8_policy, fp8_expert_layout):
    shards, tokenizer = discover_shards(src)
    stats = collections.Counter()
    examples = collections.defaultdict(list)
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                shape = tuple(f.get_slice(key).get_shape())
                fake = ShapeOnlyTensor(shape)
                kind = should_quantize_fp8(key, fake, fp8_policy, fp8_expert_layout)
                bucket = kind or ("skip_lm_head" if key.startswith("lm_head.") else "passthrough")
                stats[bucket] += 1
                if len(examples[bucket]) < 5:
                    examples[bucket].append((key, shape))
    print(f"tokenizer_json {tokenizer}")
    print("stats " + json.dumps(dict(sorted(stats.items())), sort_keys=True))
    for bucket in sorted(examples):
        print(bucket)
        for key, shape in examples[bucket]:
            print(f"  {key} {shape}")


def main():
    ap = argparse.ArgumentParser(description="Convert DiffusionGemma HF shards to ComfyUI safetensors.")
    ap.add_argument("--src", required=True, help="HuggingFace snapshot directory with model shards and tokenizer.json")
    ap.add_argument("--job", action="append", metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; PRECISION is bf16 or fp8")
    ap.add_argument("--fp8-policy", choices=("balanced", "conservative"), default="balanced",
                    help="balanced adds decoder embedding and dense MLP fp8; conservative matches the baseline footprint")
    ap.add_argument("--fp8-expert-layout", choices=("fused", "split"), default="fused",
                    help="fused keeps gate_up expert banks; split emits gate_proj/up_proj/down_proj expert banks")
    ap.add_argument("--fp8-max", type=float, default=DEFAULT_FP8_MAX,
                    help="absolute FP8 scaling divisor; default is torch.finfo(float8_e4m3fn).max")
    ap.add_argument("--dump", action="store_true", help="print tensor routing without writing outputs")
    args = ap.parse_args()

    if args.dump:
        dump(args.src, args.fp8_policy, args.fp8_expert_layout)
    if not args.job:
        if args.dump:
            return
        raise SystemExit("at least one --job is required unless --dump is set")
    for job in args.job:
        write_job(args.src, job, args.fp8_policy, args.fp8_max, args.fp8_expert_layout)


if __name__ == "__main__":
    main()
