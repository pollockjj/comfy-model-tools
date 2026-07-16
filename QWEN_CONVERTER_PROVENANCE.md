# Qwen converter provenance

## Pre-Rev0 reconstruction attempt

Commit `1c21d4d` was an unverified reconstruction attempt. It is not a release
recipe and is not a converter revision. Its measured failures established two
requirements for Rev0:

- Qwen3-VL FP8 scale is float32 `amax * float32(1.0 / 416.0)`, not
  `amax / 416.0`. The wrong operation generated
  `de5204f63b883abfdf32c54a28f173b43cc3f580f7ecb31d04a6793ba84b168a`
  instead of
  `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094`.
- Qwen3.5 release construction is model-specific. The released 2B and 4B files
  carry no safetensors metadata; the released 9B file carries
  `{"format": "pt"}`. A 9B recipe must never be generalized to 2B or 4B.

## Rev0 acceptance contract

Rev0 is the first accepted recipe. Each supported artifact has its own pinned
original-source identity, exact transformation, release metadata, and embedded
canonical output SHA256. A model is supported only after the complete artifact
is regenerated from the pinned original source and its full-file SHA256 is
identical to the artifact currently published by Comfy-Org on Hugging Face.

No passing result for one family member authorizes behavior for another family
member. Unsupported artifacts fail loud.

### Pinned original sources

| Original repository | Revision | Ordered source-shard SHA256 values |
| --- | --- | --- |
| `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | `30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9`, `046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6` |
| `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | `d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa`, `8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5`, `83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192`, `0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5` |
| `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61`, `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188` |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e`, `31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330`, `7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec`, `b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6` |

Qwen3-VL 8B NVFP4 reproduces the CUDA operation order in comfy-kitchen
`v0.2.9` (`5b45bd0`): encode values by multiplying with the reciprocal scale.
That release was compiled with `--use_fast_math`; Rev0 emulates the observed
one-ULP `rcp.approx.f32` results on CPU and then applies round-to-nearest-even
E2M1 packing. The complete output SHA gate is mandatory.

## Rev0 verification record

The complete matrix was regenerated from the pinned original sources on
2026-07-15. Every row passed the embedded complete-file SHA256 gate.

| Converter | Artifact | Canonical and generated SHA256 |
| --- | --- | --- |
| `convert_qwen3vl.py` | Qwen3-VL 4B BF16 | `36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34` |
| `convert_qwen3vl.py` | Qwen3-VL 4B FP8 scaled | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` |
| `convert_qwen3vl.py` | Qwen3-VL 8B BF16 | `68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9` |
| `convert_qwen3vl.py` | Qwen3-VL 8B FP8 scaled | `4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd` |
| `convert_qwen3vl.py` | Qwen3-VL 8B NVFP4 | `e462e9e0c3b9313ae17f82040d7c77beb92d7aef3e40692d7803228dab7c3b98` |
| `convert_qwen35.py` | Qwen3.5 2B BF16 | `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` |
| `convert_qwen35.py` | Qwen3.5 4B BF16 | `9fb3ae42003750fe2d16350259a3ec07761d6d13a8e2b244a6e22fa9d8050841` |
| `convert_qwen35.py` | Qwen3.5 9B BF16 | `7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53` |

The Qwen3-VL 8B NVFP4 gate was checked against a reference file independently
verified as `e462e9e0...`. A same-sized local file hashing `9a529749...` was
rejected as a reference before the accepted run.
