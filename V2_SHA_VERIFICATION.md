# V2 Converter SHA-Identity Verification

The V2 converters are Kijai's V1 Gemma-4 / DiffusionGemma conversions reskinned to the
SeedVR2 `--src/--job PRECISION:OUT[:SHA256]` interface. Output is byte-identical to the
shipped models; the only change from V1 is the CLI/structure. Both converters are
matmul-free (bf16 passthrough; fp8 is elementwise `amax/416`), so output is
device-independent.

End-to-end verification: each converter was run from its pinned Hugging Face snapshot, and
every produced file's sha256 was compared against the shipped/canonical value.

Verified 2026-07-03 on avenger (CPU).

| converter | job | HF source (pinned rev) | output sha256 | canonical | result |
|:--|:--|:--|:--|:--|:--|
| `convert_gemma4_v2.py` | bf16 | `google/gemma-4-E4B-it` @ `fee6332c` | `afe21e7c99d5a2ba52bc246a464d2458726204c3ce98ee81398204786ecab5ab` | `afe21e7c…` | **OK** |
| `convert_gemma4_v2.py` | fp8_scaled | `google/gemma-4-E4B-it` @ `fee6332c` | `bf0b4fa2e41a25684dc9e9b256cd505564f02fed09be3da95ce024e653e2c52b` | `bf0b4fa2…` | **OK** |
| `convert_diffusiongemma_v2.py` | bf16 | `google/diffusiongemma-26B-A4B-it` @ `0f28bc42` | `495d347e1b6c1aa13338741a17d1f5632f3ad4adb11f85f8eeb6ec026db418d1` | `495d347e…` | **OK** |
| `convert_diffusiongemma_v2.py` | fp8 | `google/diffusiongemma-26B-A4B-it` @ `0f28bc42` | `3d26c504c323bc78fa2d51dbc8433ba4ccf45dcb015b46122d2e37e4c4496015` | `3d26c504…` | **OK** |

All four jobs byte-match the shipped models. int8 convrot and the comfy-quants nvfp4/mxfp8
producers were introduced on the separate V3 track; their canonical entry points are
now `convert_gemma4.py` and `convert_diffusion_gemma.py`.
