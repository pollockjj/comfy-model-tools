# ComfyUI Model Tools

Utility scripts for packaging models for ComfyUI.

## Convert DiffusionGemma

`convert_diffusion_gemma.py` is the only DiffusionGemma converter. It exposes one
canonical recipe for each supported quant: `bf16`, `fp8`, `int8`, `int4`, `mxfp8`,
and `nvfp4`. Every recipe has one fixed accepted SHA256 embedded in the script and
fails if its output differs by one byte. Structural QKV fusion and expert layouts
are internal parts of the canonical recipes, not separate public jobs.

```sh
python convert_diffusion_gemma.py --src /path/to/source \
    --job bf16:/path/to/diffusiongemma_bf16.safetensors \
    --job fp8:/path/to/diffusiongemma_fp8.safetensors \
    --job int8:/path/to/diffusiongemma_int8_convrot.safetensors \
    --job int4:/path/to/diffusiongemma_int4_convrot.safetensors \
    --job mxfp8:/path/to/diffusiongemma_mxfp8.safetensors \
    --job nvfp4:/path/to/diffusiongemma_nvfp4.safetensors
```

## Convert Gemma 4 E2B / E4B

`convert_gemma4.py` converts either Gemma 4 variant from a Hugging Face snapshot or
an existing ComfyUI BF16 text encoder. Each job consumes and releases its source tensors
so BF16, scaled FP8, and INT8 ConvRot conversion fit within 32 GiB host memory. The INT8 policy covers eligible
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
