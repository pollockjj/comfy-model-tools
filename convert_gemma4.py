"""Convert a Gemma 4 (E2B/E4B) HF checkpoint into the ComfyUI text-encoder layout.

    python convert_gemma4.py HF_DIR OUT.safetensors [--sha256 EXPECTED]

HF_DIR must hold model.safetensors (single-file HF export, bf16) and tokenizer.json.

Key mapping (verified by exhaustive set-diff against a known-good ComfyUI file):
  model.language_model.*  -> model.*
  model.audio_tower.*     -> audio_model.*
  model.vision_tower.*    -> vision_model.*
  model.embed_audio.embedding_projection  -> audio_projector.embedding_projection
  model.embed_vision.embedding_projection -> multi_modal_projector.embedding_projection
QAT observer tensors (input/output min/max) pass through unchanged — the ComfyUI
layout retains them.

KV-shared layers (num_kv_shared_layers tail of the stack) have no k_proj/v_proj/k_norm
in HF exports. ComfyUI instantiates those modules on every layer (unused at runtime on
shared layers), so the converter fills each shared layer's slots from the layer it
shares KV with: the last non-shared layer of the same attention type per
config.text_config.layer_types.

tokenizer.json is embedded verbatim as the uint8 tensor `tokenizer_json`.
"""
# ruff: noqa: T201
import argparse
import hashlib
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX_MAP = [
    ("model.language_model.", "model."),
    ("model.audio_tower.", "audio_model."),
    ("model.vision_tower.", "vision_model."),
    ("model.embed_audio.", "audio_projector."),
    ("model.embed_vision.", "multi_modal_projector."),
]


def map_key(key):
    for src, dst in PREFIX_MAP:
        if key.startswith(src):
            return dst + key[len(src):]
    raise ValueError(f"unmapped HF key: {key}")


def kv_share_sources(cfg):
    tc = cfg.get("text_config", cfg)
    n = tc["num_hidden_layers"]
    shared = tc["num_kv_shared_layers"]
    types = tc["layer_types"]
    first_shared = n - shared
    last_of_type = {}
    for i in range(first_shared):
        last_of_type[types[i]] = i
    return {i: last_of_type[types[i]] for i in range(first_shared, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("dst")
    ap.add_argument("--sha256", default=None, help="expected sha256 of the written file")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.hf_dir, "config.json")))
    src_path = os.path.join(args.hf_dir, "model.safetensors")
    tok_path = os.path.join(args.hf_dir, "tokenizer.json")

    out = {}
    with safe_open(src_path, framework="pt", device="cpu") as st:
        for key in st.keys():
            t = st.get_tensor(key)
            if t.dtype != torch.bfloat16 and t.is_floating_point():
                t = t.to(torch.bfloat16)
            out[map_key(key)] = t

    filled = 0
    for shared_idx, src_idx in kv_share_sources(cfg).items():
        for leaf in ("self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_norm.weight"):
            dst_key = f"model.layers.{shared_idx}.{leaf}"
            src_key = f"model.layers.{src_idx}.{leaf}"
            if dst_key not in out:
                out[dst_key] = out[src_key].clone()
                filled += 1
    print(f"filled {filled} kv-shared slots from source layers")

    tok = open(tok_path, "rb").read()
    out["tokenizer_json"] = torch.tensor(list(tok), dtype=torch.uint8)
    print(f"embedded tokenizer_json ({len(tok)} bytes)")

    save_file(out, args.dst)
    print(f"wrote {len(out)} tensors -> {args.dst}")

    sha = hashlib.sha256()
    with open(args.dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    print(f"sha256 {digest}")
    if args.sha256 and digest != args.sha256:
        raise SystemExit(f"sha256 mismatch: expected {args.sha256}")


if __name__ == "__main__":
    main()
