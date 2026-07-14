# ComfyUI Model Tools

Utility scripts for packaging models for ComfyUI.

## Convert DiffusionGemma

`convert_diffusion_gemma.py` converts a DiffusionGemma HF snapshot or ComfyUI BF16
text encoder with repeatable `--job` arguments. It supports `bf16`, `fp8`, full-map
`int8` ConvRot, packed-W4 `int4`, fused-bank `mxfp8_fused`, fused-bank plus fused-attention
`mxfp8_fused_qkv`, and the payload-preserving `mxfp8_qkv_patch`. The MXFP8 jobs
quantize the tied decoder token embedding as well as decoder matrices and expert
banks. The `int4` job stores only the routed-expert banks as W4A4. Every eligible
nonexpert decoder linear and the tied embedding remain INT8 ConvRot W8A8.

```sh
python convert_diffusion_gemma.py --src /path/to/source \
    --job bf16:/path/to/diffusiongemma_bf16.safetensors \
    --job fp8:/path/to/diffusiongemma_fp8.safetensors \
    --job int8:/path/to/diffusiongemma_int8_convrot.safetensors \
    --job int4:/path/to/diffusiongemma_int4_convrot.safetensors \
    --job mxfp8_fused_qkv:/path/to/diffusiongemma_mxfp8.safetensors
```

## Convert DiffusionGemma NVFP4

`convert_diffusion_gemma_block_formats.py` converts an existing ComfyUI BF16
DiffusionGemma text encoder to MXFP8 or NVFP4 through ComfyUI's comfy-kitchen-backed
tensor-core layouts. The `nvfp4_fused` job transcodes NVIDIA's calibrated NVFP4
expert payload without requantization. The legacy `nvfp4` job preserves the frozen
rev-0 storage baseline: it creates a runnable model, but contains no
DiffusionGemma-specific throughput optimization.

```sh
COMFYUI_ROOT=/path/to/ComfyUI CUDA_VISIBLE_DEVICES=0 python \
    convert_diffusion_gemma_block_formats.py \
    --src /path/to/diffusiongemma_bf16.safetensors \
    --nvidia-model-dir /path/to/nvidia/diffusiongemma_nvfp4 \
    --format nvfp4_fused
```

See [DIFFUSIONGEMMA_NVFP4_REV0.md](DIFFUSIONGEMMA_NVFP4_REV0.md) for converter and
artifact hashes, tensor coverage, the measured RTX 5090 smoke result, and known limits.

## Convert Gemma 4 E2B / E4B

`convert_gemma4.py` converts either Gemma 4 variant from a Hugging Face snapshot or
an existing ComfyUI BF16 text encoder. One source load can write deterministic BF16,
Kijai-compatible scaled FP8, and INT8 ConvRot jobs. The INT8 policy covers eligible
language-model matrices while leaving vision, audio, and projector tensors in BF16.

```sh
python convert_gemma4.py --src /path/to/gemma-4-E4B-it \
    --job bf16:/path/to/gemma4_e4b_it_bf16.safetensors \
    --job fp8_scaled:/path/to/gemma4_e4b_it_fp8_scaled.safetensors

CUDA_VISIBLE_DEVICES= python convert_gemma4.py --device cpu \
    --src /path/to/gemma4_e4b_it_bf16.safetensors \
    --job int8:/path/to/gemma4_e4b_it_int8_convrot.safetensors
```

The same commands accept E2B sources. Gemma 4 NVFP4 and MXFP8 artifact contracts
remain open work; generic block-format writers are not substitutes for a validated
model-specific conversion policy. `convert_gemma4_mtp.py` separately extracts the
E2B/E4B MTP drafter sidecar from an official assistant safetensors or llama.cpp GGUF.

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
