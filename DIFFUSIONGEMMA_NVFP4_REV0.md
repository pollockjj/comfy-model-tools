# DiffusionGemma NVFP4 rev-0

**Status: RUNNABLE / UNOPTIMIZED / STORAGE BASELINE**

Rev-0 preserves the first working DiffusionGemma-26B-A4B NVFP4 conversion. It proves
that the BF16 text encoder can be converted, loaded, and used to produce the expected
answer on one smoke input. It does not contain or claim a DiffusionGemma-specific
NVFP4 kernel optimization.

## Converter identity

| Item | Value |
|:--|:--|
| Recovered working-first source SHA256 | `a033cb9ee0c73290b33b50a7542100d1a76cf17fb0be4ba1c821f123a764e45b` |
| Recovered source size | `8,153` bytes / `214` lines |
| Public portable source | `convert_diffusion_gemma.py --job nvfp4:OUT` |
| Public portable source SHA256 | `f19794a9cb6effda63c6eafffa0add3e3dd50f9f9d3160f3ba34dbe98b0bab00` |
| Converter commit | `55ca0b82eac53773a9e548b240737e1847f8304c` |

The public source differs from the recovered source in exactly two source edits:

1. `import os` was added.
2. The host-specific ComfyUI root was replaced by `COMFYUI_ROOT`, defaulting to
   `~/dev_master/ComfyUI`.

No quantization predicate, tensor transform, marker, layout call, or output write path
logic changed. On the original host, the portable default resolves to the same ComfyUI
checkout, so the generated safetensors content is unchanged.

## Conversion contract

- Input is the existing ComfyUI BF16 DiffusionGemma text-encoder safetensors.
- `TensorCoreNVFP4Layout` supplies NVFP4 quantization and serialization tensors.
- Each rank-3 expert bank is quantized expert-by-expert as 2-D matrices.
- Fused `gate_up` expert banks are split into `gate` and `up`; `down` remains separate.
- Expert weights are emitted as split/unfused banks with `num_experts: 128`.
- Decoder attention q/k/v/o and dense MLP gate/up/down 2-D weights are quantized.
- Every quant marker sets `full_precision_matrix_mult: false`.
- Other floating-point tensors pass through as BF16.

The converter also retains the recovered MXFP8 path. The NVFP4 path was unchanged by
the later MXFP8 E8M0 serialization correction.

## Reproduction

The historical working-first run selected both formats. The NVFP4-only equivalent is:

```sh
COMFYUI_ROOT=/path/to/ComfyUI \
CUDA_VISIBLE_DEVICES=0 \
/path/to/python convert_diffusion_gemma.py \
    --src /path/to/diffusiongemma_comfy_bf16.safetensors \
    --job nvfp4:/path/to/diffusiongemma_comfy_nvfp4_working.safetensors
```

Requirements are a CUDA-capable PyTorch environment, `safetensors`, a ComfyUI checkout
at `COMFYUI_ROOT`, and a comfy-kitchen backend exposed through `comfy.quant_ops`.

## Artifact audit

| Item | BF16 source | NVFP4 rev-0 |
|:--|--:|--:|
| SHA256 | `495d347e1b6c1aa13338741a17d1f5632f3ad4adb11f85f8eeb6ec026db418d1` | `a07a8cdacd46fb106a718bf07a628645f078c306eb7e2097dfcff4b5f7677cdc` |
| Bytes | `51,679,872,682` | `16,572,299,245` |
| GiB | `48.13063` | `15.434156` |

Rev-0 removes `35,107,573,437` bytes, or `67.932778%`, from the BF16 source artifact.

The NVFP4 safetensors contains `1,963` tensors:

| Stored dtype | Tensor count |
|:--|--:|
| BF16 | 782 |
| F32 | 295 |
| F8_E4M3 | 295 |
| U8 | 591 |

Its `295` NVFP4 markers cover `90` split/unfused expert banks and `205` decoder 2-D
matrices. All `295` markers set `full_precision_matrix_mult: false`; the expert markers
declare `128` experts.

## One-row execution proof

The recorded smoke run used:

- RTX 5090
- ComfyUI commit `6cf5be53e24e6697bc052f459d38ddafb4c39795`
- comfy-kitchen `0.2.15`, commit `459ef464b50024583cbf247f8719ef0883fab8d0`
- `15,727 MB` staged model memory

Measured result:

| Observation | Value |
|:--|:--|
| Expected answer | `A` |
| Produced answer | `A` |
| Triage | `No triage items found.` |
| Final generation progress | `605/32768` at `7.66 it/s` |
| Prompt wall time | `80.78 seconds` |
| Whole-prompt derived rate | `605 / 80.78 = 7.4895 tokens/second` |

The derived rate includes the full prompt interval and is not a steady-state-only kernel
measurement. A historical ten-row NVFP4 run was started, but no completed ten-row
harness artifact exists; it is not baseline evidence.

## Rev-0 limits

- The converter creates storage and metadata; it does not introduce runtime kernels.
- The expert representation is deliberately split/unfused and bypasses grouped bank BMM.
- No kernel trace exists for rev-0, so the slow path's dominant mechanism is unproven.
- No descriptor cache, static activation-scale path, fused routed-MoE kernel, or
  DiffusionGemma-specific dispatcher optimization was added.
- The one-row result establishes load/run/answer behavior only. Throughput optimization
  starts from this measured baseline and requires same-model, same-workflow, same-GPU A/B
  profiling.
