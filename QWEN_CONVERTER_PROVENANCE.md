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
