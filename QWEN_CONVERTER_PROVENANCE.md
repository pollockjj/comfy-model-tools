# Qwen converter provenance

## Rev0: historical reconstruction, rejected as a family release recipe

Commit `1c21d4d` introduced the first reconstructed Qwen conversion scripts.
The scripts are recorded as Rev0 because their exact behavior is part of the
artifact provenance even where that behavior is wrong. A passing subset does
not make Rev0 a correct family recipe.

### Measured Rev0 results

| Artifact | Canonical SHA256 | Rev0 SHA256 | Result |
| --- | --- | --- | --- |
| Qwen3-VL 4B BF16 | `36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34` | `36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34` | exact |
| Qwen3-VL 4B FP8 scaled | `54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094` | `de5204f63b883abfdf32c54a28f173b43cc3f580f7ecb31d04a6793ba84b168a` | rejected |
| Qwen3.5 2B BF16 | `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` | `878fd1eb88e97daaa6d235d5c25bdec0271d3a031f085eb51876e18d7ae50a59` | rejected |
| Qwen3.5 9B BF16 | `7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53` | `7e6e9f08d598f829cb940e60ac0c698e1f1c27a47daffd7e598cd78c78b4cc53` | exact for 9B only |

The Qwen3-VL FP8 defect is arithmetic, not serialization. For every one of the
252 released 4B scale tensors, the canonical scale is computed as float32
`amax * float32(1.0 / 416.0)`. Rev0 uses `amax / 416.0`; 115 of 252 scale
tensors differ by one float32 ULP, and their associated FP8 weights differ.

The Qwen3.5 defect is release metadata. Rev0 always saves with
`{"format": "pt"}`. The released 2B and 4B files have no safetensors metadata;
the released 9B file has `{"format": "pt"}`. This is why Rev0 reproduces 9B
but cannot serve as the family converter.

Rev1 must retain none of these defects. It is accepted only after each artifact
it claims is regenerated in full and its complete-file SHA256 is identical to
the canonical Hugging Face artifact.

## Rev1

Rev1 starts from exact pinned original-source shard SHA256 values, uses
variant-defined release metadata, embeds every canonical output SHA256, and
fails if the source, requested artifact, caller-supplied SHA, or generated file
does not match that provenance. Qwen3-VL FP8 uses float32 reciprocal
multiplication, not Rev0 division.
