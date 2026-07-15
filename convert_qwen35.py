#!/usr/bin/env python3
"""
Convert Qwen3.5 checkpoints to ComfyUI text-encoder safetensors.

REV0 HISTORICAL PROVENANCE ONLY. DO NOT USE THIS AS A FAMILY RELEASE RECIPE.

This file records the first attempted reconstruction committed as 1c21d4d. It
reproduces the released Qwen3.5-9B SHA256 exactly, but it always writes
``{"format": "pt"}``. That makes its 2B output wrong and makes the 4B path wrong
by the same release-metadata mismatch. Observed 2B output:
878fd1eb88e97daaa6d235d5c25bdec0271d3a031f085eb51876e18d7ae50a59;
canonical 2B:
aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1.
Observed 9B output and canonical SHA256:
7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53.
This behavior is retained unchanged for posterity and provenance. Rev1 will
encode and verify each released family member explicitly.

The attempted conversion preserves the released 9B dtype policy:

Example:
  python convert_qwen35.py --src qwen3.5_9b_bf16.safetensors \
      --job bf16:qwen3.5_9b_bf16.safetensors

A job may carry an expected SHA256 (PRECISION:OUT:SHA256) to verify the written
file. Source may be an existing released/ComfyUI bf16 safetensors or an HF
snapshot directory containing model.safetensors or model-*-of-*.safetensors.
"""
import argparse
import glob
import hashlib
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


F32_SUFFIXES = (".linear_attn.A_log", ".linear_attn.norm.weight")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


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
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as st:
            for key in st.keys():
                out[key] = st.get_tensor(key)
    print(f"loaded Qwen3.5 source: {len(out)} tensors, {len(files)} file(s)")
    return out


def cast(sd, precision):
    if precision != "bf16":
        raise SystemExit(f"unknown precision: {precision}")
    out = {}
    kept_f32 = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if v.is_floating_point():
            if k.endswith(F32_SUFFIXES):
                out[k] = v.to(torch.float32)
                kept_f32 += 1
            else:
                out[k] = v.to(torch.bfloat16)
        else:
            out[k] = v
    print(f"bf16: kept_f32={kept_f32}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Qwen3.5 Rev0 historical reconstruction; valid for the 9B artifact only.")
    ap.add_argument("--src", required=True, help="HF snapshot dir or ComfyUI/released bf16 safetensors")
    ap.add_argument("--job", action="append", required=True, metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; one source load serves every job")
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
        tensors = cast(base, precision)
        save_file(tensors, out, metadata={"format": "pt"})
        digest = sha256(out)
        verdict = "" if expected is None else ("  OK" if digest == expected else "  MISMATCH")
        print(f"{precision:14s} {digest}  {out}{verdict}")
        if expected is not None and digest != expected:
            mismatched.append(out)

    if mismatched:
        raise SystemExit(f"SHA256 mismatch: {', '.join(mismatched)}")


if __name__ == "__main__":
    main()
