# ComfyUI Model Tools

Utility scripts for packaging models for ComfyUI.

`merge_safetensors.py`: Merge all safetensors files in a directory into a single safetensors file. If duplicate keys exist, they are skipped with a warning:

```sh
python.exe merge_safetensors.py ".\source\folder\path" "target_file_path.safetensors"
```

`seedvr2_convert.py`: Convert a ByteDance SeedVR2 `.pth` checkpoint into ComfyUI-loadable safetensors in `fp16`, `fp8_e4m3fn`, `fp8_e4m3fn_mixed_block35_fp16`, `mxfp8`, or `nvfp4`. Each `--job` writes one precision; an optional `PRECISION:OUT:SHA256` verifies the written file:

```sh
python seedvr2_convert.py --src seedvr2_ema_3b.pth --cond pos_emb.pt,neg_emb.pt \
    --job fp16:seedvr2_3b_fp16.safetensors \
    --job nvfp4:seedvr2_3b_nvfp4.safetensors
```

The `mxfp8` and `nvfp4` modes require the `comfy-kitchen` package; the script exits with a message naming it if it is not installed.

`convert_diffusion_gemma.py`: Convert a Google DiffusionGemma HuggingFace snapshot into ComfyUI text-encoder safetensors. `bf16` writes Comfy key names without quantization. `fp8` writes Comfy-native in-band `weight` / `weight_scale` / `comfy_quant` triplets:

```sh
python convert_diffusion_gemma.py --src diffusiongemma-26B-A4B-it \
    --job bf16:diffusiongemma_comfy_bf16.safetensors \
    --job fp8:diffusiongemma_comfy_fp8.safetensors
```

The default `--fp8-policy balanced` quantizes fused MoE expert banks, decoder q/o attention weights, decoder dense MLP weights, and the decoder token embedding. `--fp8-policy conservative` keeps the original baseline footprint: fused MoE expert banks plus decoder q/o attention weights only.

`convert_gemma4.py`: Convert Google Gemma 4 safetensors into ComfyUI format or reproduce the published Comfy-Org FP8-scaled E4B artifact from the Comfy BF16 file. `fp8_scaled` matches Kijai's rev-1 policy: large 2D `model.*.weight` tensors use `float8_e4m3fn` plus scalar `weight_scale` and `comfy_quant`; all other tensors pass through unchanged.

```sh
python convert_gemma4.py --src gemma4_e4b_it_bf16.safetensors \
    --job fp8_scaled:gemma4_e4b_it_fp8_scaled.safetensors:bf0b4fa2e41a25684dc9e9b256cd505564f02fed09be3da95ce024e653e2c52b
```
