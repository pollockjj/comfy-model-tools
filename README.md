# ComfyUI Model Tools

Utility scripts for packaging models for ComfyUI.

## Merge Safetensors

`merge_safetensors.py`: Merge all safetensors files in a directory into a single safetensors file. If duplicate keys exist, they are skipped with a warning:

```sh
python.exe merge_safetensors.py ".\source\folder\path" "target_file_path.safetensors"
```

## Convert SeedVR2 to Safetensors

`seedvr2_convert.py`: Convert a ByteDance SeedVR2 `.pth` checkpoint into ComfyUI-loadable safetensors in `fp16`, `fp8_e4m3fn`, `mxfp8`, `nvfp4`, or `int8`. Each `--job` writes one precision; an optional `PRECISION:OUT:SHA256` verifies the written file:

```sh
python seedvr2_convert.py --src seedvr2_ema_3b.pth --cond pos_emb.pt,neg_emb.pt \
    --job fp16:seedvr2_3b_fp16.safetensors \
    --job nvfp4:seedvr2_3b_nvfp4.safetensors
```

The `mxfp8`, `nvfp4`, and `int8` modes require the `comfy-kitchen` package; the script exits with a message naming it if it is not installed.

## Reproduce Qwen release artifacts

`convert_qwen3vl.py` and `convert_qwen35.py` are Rev0 deterministic release
recipes. They accept only embedded, SHA-pinned original Qwen checkpoints and
always verify the complete output file against the corresponding artifact
published by Comfy-Org on Hugging Face. Unsupported models or source revisions
fail instead of inheriting a recipe from another family member.

Qwen3-VL supports the released 4B BF16/FP8-scaled artifacts and the released 8B
BF16/FP8-scaled/NVFP4 artifacts:

```sh
python convert_qwen3vl.py --src Qwen3-VL-4B-Instruct-snapshot \
    --job bf16:qwen3vl_4b_bf16.safetensors \
    --job fp8_scaled:qwen3vl_4b_fp8_scaled.safetensors
```

Qwen3.5 selects exactly one model-specific recipe from the pinned source shard
hashes: 2B is copied byte-for-byte, 4B is merged without metadata, and 9B is
merged with `{"format": "pt"}` metadata.

```sh
python convert_qwen35.py --src Qwen3.5-4B-snapshot \
    --out qwen3.5_4b_bf16.safetensors
```

See [QWEN_CONVERTER_PROVENANCE.md](QWEN_CONVERTER_PROVENANCE.md) for the pinned
source revisions, failed pre-Rev0 reconstruction findings, and all eight
canonical output hashes proved by full regeneration.

## Quantize to `int8-convrot`

Dry-run first (good idea to do this on a new architecture) — prints the plan, writes nothing
  
`python quant_int8_auto.py model_bf16.safetensors --dry-run`

Quantize (defaults: absmax, min-gemm 256)
  
`python quant_int8_auto.py model_bf16.safetensors model_int8_convrot.safetensors`

### args

```
--dry-run — plan only (grouped quantize list + what's left behind, with reasons)
--exclude RE / --include RE — regex overrides on layer base names
--min-gemm N — skip layers with min(N,K) < N (default 256)
--mseclip — MSE-optimal clip instead of absmax, usually lower weight error, experimental
--downcast-fp32 — shrink stray fp32 passthrough to the compute dtype
--warn-thresh F — warn on any quantized layer over F% relerr (default 2.0)
--verify-report PATH — dump the full per-layer (relerr, cos, gs) table
```
