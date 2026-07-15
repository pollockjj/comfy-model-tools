#!/usr/bin/env python3
"""
Reproduce the released Comfy-Org Qwen3.5 BF16 text-encoder artifacts.

Rev0 is model-specific. It accepts only these pinned original checkpoints:

  Qwen/Qwen3.5-2B @ 15852e8c16360a2fea060d615a32b45270f8a8fc
  Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a

Each model has an explicit released construction:

  2B  The official single safetensors file is already byte-identical to the
      Comfy-Org artifact. Copy it without reserialization or metadata changes.
  4B  Merge the two official shards unchanged with no safetensors metadata.
  9B  Merge the four official shards unchanged with {"format": "pt"} metadata.

The source shard hashes and complete output hashes are embedded. Any source or
output drift fails loud. No recipe for one model is applied to another model.

Example:
  python convert_qwen35.py --src Qwen3.5-4B-snapshot \
      --out qwen3.5_4b_bf16.safetensors
"""
import argparse
import glob
import hashlib
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


PINNED_SOURCE_SHAS = {
    "qwen3.5_2b": (
        "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
    ),
    "qwen3.5_4b": (
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
    ),
    "qwen3.5_9b": (
        "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e",
        "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330",
        "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec",
        "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6",
    ),
}

EXPECTED_SHAS = {
    "qwen3.5_2b": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
    "qwen3.5_4b": "9fb3ae42003750fe2d16350259a3ec07761d6d13a8e2b244a6e22fa9d8050841",
    "qwen3.5_9b": "7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53",
}

RELEASE_METADATA = {
    "qwen3.5_4b": None,
    "qwen3.5_9b": {"format": "pt"},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_safetensors(src):
    if os.path.isfile(src):
        return [src]
    patterns = (
        "model.safetensors",
        "model.safetensors-*-of-*.safetensors",
        "model-*-of-*.safetensors",
    )
    for pattern in patterns:
        files = sorted(glob.glob(os.path.join(src, pattern)))
        if files:
            return files
    raise SystemExit(f"no original Qwen3.5 safetensors source found in {src}")


def identify_pinned_source(files):
    digests = tuple(sha256(file) for file in files)
    for variant, expected in PINNED_SOURCE_SHAS.items():
        if digests == expected:
            print(f"verified pinned source: {variant}")
            return variant
    raise SystemExit(f"unrecognized source shard SHA256 tuple: {digests}")


def merge_shards(files):
    tensors = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as st:
            for key in st.keys():
                if key in tensors:
                    raise SystemExit(f"duplicate tensor across source shards: {key}")
                tensors[key] = st.get_tensor(key)
    print(f"loaded {len(tensors)} tensors from {len(files)} pinned source shards")
    return tensors


def main():
    ap = argparse.ArgumentParser(description="Qwen3.5 Rev0 deterministic release converter.")
    ap.add_argument("--src", required=True, help="pinned original HF snapshot directory")
    ap.add_argument("--out", required=True, help="output ComfyUI safetensors file")
    args = ap.parse_args()

    files = iter_safetensors(args.src)
    variant = identify_pinned_source(files)
    if variant == "qwen3.5_2b":
        if os.path.abspath(files[0]) != os.path.abspath(args.out):
            shutil.copyfile(files[0], args.out)
    else:
        tensors = merge_shards(files)
        save_file(tensors, args.out, metadata=RELEASE_METADATA[variant])

    digest = sha256(args.out)
    expected = EXPECTED_SHAS[variant]
    verdict = "OK" if digest == expected else "MISMATCH"
    print(f"{variant:12s} {digest}  {args.out}  {verdict}")
    if digest != expected:
        raise SystemExit(f"SHA256 mismatch for {variant}: expected {expected}, got {digest}")


if __name__ == "__main__":
    main()
