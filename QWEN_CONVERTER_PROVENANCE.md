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
