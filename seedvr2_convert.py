#!/usr/bin/env python3
"""
Convert ByteDance SeedVR2 checkpoints to ComfyUI safetensors.

Each source tensor is converted to the target precision and written to safetensors with
the original key names (safetensors sorts keys; no metadata). NVFP4 jobs emit native
ComfyUI quantized-weight keys for selected 2D Linear weights. DiT files additionally
embed the fixed text conditioning (--cond) as positive_conditioning / negative_conditioning,
copied through as-is (bf16).

Precisions:
  fp16                          every tensor -> float16
  fp8_e4m3fn                    every tensor -> float8_e4m3fn
  nvfp4                         eligible 2D .weight tensors -> TensorCoreNVFP4Layout; high-risk
                                input/output projections, text/embedding input layers, and
                                tensorcore-ineligible shapes stay float16; everything else ->
                                float16. All DiT blocks are quantized
  mxfp8                         eligible 2D .weight tensors -> TensorCoreMXFP8Layout; high-risk
                                input/output projections and text/embedding input layers stay
                                float16; everything else -> float16. All DiT blocks are quantized
  int8                          eligible 2D .weight tensors -> TensorWiseINT8Layout with ConvRot
                                (per-channel weight scale + online group-wise Hadamard activation
                                rotation, groupsize 256); high-risk input/output projections,
                                text/embedding input layers, and weights whose in_features is not
                                a multiple of 256 stay float16; everything else -> float16.
                                All DiT blocks are quantized

Examples:
  # 3B DiT -> fp16 and fp8, conditioning baked in (one load serves both jobs)
  python seedvr2_convert.py --src seedvr2_ema_3b.pth --cond pos_emb.pt,neg_emb.pt \
      --job fp16:seedvr2_3b_fp16.safetensors \
      --job fp8_e4m3fn:seedvr2_3b_fp8_e4m3fn.safetensors

  # 7B DiT -> fp16 and fp8
  python seedvr2_convert.py --src seedvr2_ema_7b.pth --cond pos_emb.pt,neg_emb.pt \
      --job fp16:seedvr2_7b_fp16.safetensors \
      --job fp8_e4m3fn:seedvr2_7b_fp8_e4m3fn.safetensors

  # VAE (no conditioning)
  python seedvr2_convert.py --src ema_vae.pth --job fp16:ema_vae_fp16.safetensors

  # 3B DiT -> NVFP4 and MXFP8 (ComfyUI-native quantized weights), conditioning baked in
  python seedvr2_convert.py --src seedvr2_ema_3b.pth --cond pos_emb.pt,neg_emb.pt \
      --job nvfp4:seedvr2_3b_nvfp4.safetensors \
      --job mxfp8:seedvr2_3b_mxfp8.safetensors

  # 7B DiT -> NVFP4 and MXFP8
  python seedvr2_convert.py --src seedvr2_ema_7b.pth --cond pos_emb.pt,neg_emb.pt \
      --job nvfp4:seedvr2_7b_nvfp4.safetensors \
      --job mxfp8:seedvr2_7b_mxfp8.safetensors

  # 3B DiT -> INT8 ConvRot (ComfyUI-native quantized weights), conditioning baked in
  python seedvr2_convert.py --src seedvr2_ema_3b.pth --cond pos_emb.pt,neg_emb.pt \
      --job int8:seedvr2_3b_int8.safetensors

  # 7B DiT -> INT8 ConvRot
  python seedvr2_convert.py --src seedvr2_ema_7b.pth --cond pos_emb.pt,neg_emb.pt \
      --job int8:seedvr2_7b_int8.safetensors

A job may carry an expected SHA256 (PRECISION:OUT:SHA256) to verify the written file.

==========================================================================================
Provenance
==========================================================================================
Source checkpoints (Apache-2.0), pinned to the exact HuggingFace revision converted from:

  ByteDance-Seed/SeedVR2-3B  @ 37255ff8cccfb01071b87f635a5948ca8d53117c
  https://huggingface.co/ByteDance-Seed/SeedVR2-3B/tree/37255ff8cccfb01071b87f635a5948ca8d53117c
    6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b  seedvr2_ema_3b.pth   (2025-06-22 "update ckpt")
    c7df8a67e68b7f9aca3d5d2153d2ce8ab4373687741a0f9ce87cb356ace51cac  ema_vae.pth
    fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799  pos_emb.pt
    6a43e5800ef2354f1c156d27535834da055cbec8248298b8923492bba2076581  neg_emb.pt

  ByteDance-Seed/SeedVR2-7B  @ eb0c4281d41ba3767d4f14370f0e37e9e9180c16
  https://huggingface.co/ByteDance-Seed/SeedVR2-7B/tree/eb0c4281d41ba3767d4f14370f0e37e9e9180c16
    e1b2ae25505607e61f2a7dc7967ba778aaf3e3626d9969ce6e24c52d9ddebfcd  seedvr2_ema_7b.pth
    ced5706c976d5879efcab9e108349d67abcbd8a9b36a1f48bf0f19c24164a264  seedvr2_ema_7b_sharp.pth

Conditioning embedded in every DiT output (from the 3B repo above):
    positive_conditioning  <-  pos_emb.pt  (fa07a148...)
    negative_conditioning  <-  neg_emb.pt  (6a43e580...)

Outputs  ( sha256  file  <-  source.pth, precision [+ conditioning] ):
  20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1  ema_vae_fp16.safetensors  <- ema_vae.pth, fp16 (no cond)
  98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660  seedvr2_3b_fp16.safetensors  <- seedvr2_ema_3b.pth, fp16 + cond
  a0226eaa2c3e6f47ae5ce83225120f16479da890ced1a3bc32b1a14619787914  seedvr2_3b_fp8_e4m3fn.safetensors  <- seedvr2_ema_3b.pth, fp8_e4m3fn + cond
  c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157  seedvr2_3b_int8.safetensors  <- seedvr2_ema_3b.pth, int8 + cond
  768623e3bfb1752b4d0668782751b5fead58b1bcb153f0b5e03a423095630297  seedvr2_3b_mxfp8.safetensors  <- seedvr2_ema_3b.pth, mxfp8 + cond
  c8dea38b04d43295621726e2cd371c0d2d001006169c113aea17950f2cb2e295  seedvr2_3b_nvfp4.safetensors  <- seedvr2_ema_3b.pth, nvfp4 + cond
  2742ca6fee63bc5cc1773f426dd4b07b78cad27f51c9ea5cd42b035e6b592252  seedvr2_7b_fp16.safetensors  <- seedvr2_ema_7b.pth, fp16 + cond
  5065e77d647dd553d9090a81e20d6de590d931a61df79d785e008433926ee418  seedvr2_7b_fp8_e4m3fn.safetensors  <- seedvr2_ema_7b.pth, fp8_e4m3fn + cond
  5aa0d25fc9d35e449b659d0c9a5dcb22e2a4fa04032101b95a39da42b32c1be6  seedvr2_7b_int8.safetensors  <- seedvr2_ema_7b.pth, int8 + cond
  b40804f47910d96c5089c728cc7ec8b57b956750eabb6397dc4e6e697477263d  seedvr2_7b_mxfp8.safetensors  <- seedvr2_ema_7b.pth, mxfp8 + cond
  cc4af1a7bd5377066496f393555478323e806fa21163bdbe3409451aface9b93  seedvr2_7b_nvfp4.safetensors  <- seedvr2_ema_7b.pth, nvfp4 + cond
  70823bca54b9c24eeb56e1c452697c7c2a430867e58db0e376c6e260f3a4489d  seedvr2_7b_sharp_fp16.safetensors  <- seedvr2_ema_7b_sharp.pth, fp16 + cond
  7602c5f70868d28e7730035e4e9d745b05d661c8f0a7eb758e63f9c8603596ef  seedvr2_7b_sharp_fp8_e4m3fn.safetensors  <- seedvr2_ema_7b_sharp.pth, fp8_e4m3fn + cond
  db48be2f1cc7e36b01a2aa529810f5d9c6a971edd29be225cf1b0eb18d51c366  seedvr2_7b_sharp_int8.safetensors  <- seedvr2_ema_7b_sharp.pth, int8 + cond
  0d621ec1561a11ca9b5f432ec6d4e09b263b61f4b83b0280552c8b4add030ec3  seedvr2_7b_sharp_mxfp8.safetensors  <- seedvr2_ema_7b_sharp.pth, mxfp8 + cond
  80d57af7722f5a5bd4c01d2ab2688f2bf05e552e59d3d3287257de709db10397  seedvr2_7b_sharp_nvfp4.safetensors  <- seedvr2_ema_7b_sharp.pth, nvfp4 + cond
"""
import argparse
import collections
import hashlib
import json

import torch
from safetensors.torch import save_file

# int8 quantization is delegated to comfy-quants' canonical stock-ComfyUI producer
# (byte-matches ComfyUI's own save path), not comfy-kitchen from_float.
from comfy_quants.backends.int8_tensorwise_model_export import _quantize_int8_tensorwise_per_row
from comfy_quants.formats.int8_tensorwise import int8_tensorwise_checkpoint_quant_config

FP8 = torch.float8_e4m3fn
NVFP4_LAYOUT = "TensorCoreNVFP4Layout"
NVFP4_ALIGNMENT = 32
MXFP8_LAYOUT = "TensorCoreMXFP8Layout"
INT8_LAYOUT = "TensorWiseINT8Layout"
INT8_FORMAT = "int8_tensorwise"
INT8_CONVROT_GROUPSIZE = 256  # power-of-4 Hadamard group size; in_features must be a multiple
NVFP4_HIGH_RISK_PREFIXES = (
    "emb_in.",
    "txt_in.",
    "vid_in.",
    "vid_out.",
)
MXFP8_HIGH_RISK_PREFIXES = NVFP4_HIGH_RISK_PREFIXES
INT8_HIGH_RISK_PREFIXES = NVFP4_HIGH_RISK_PREFIXES


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state_dict(pth):
    obj = torch.load(pth, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(obj, dict):
        for key in ("state_dict", "ema", "model", "module", "params", "ema_model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if any(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise SystemExit(f"Unrecognized checkpoint structure: {type(obj)}")


def comfy_quant_tensor(format_name, extra=None):
    conf = {"format": format_name}
    if extra:
        conf.update(extra)
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def roundup(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def nvfp4_tensorcore_eligible(v):
    if v.dim() != 2:
        return False
    return roundup(v.shape[1], 16) % NVFP4_ALIGNMENT == 0


def should_quantize_nvfp4(k, v):
    if not k.endswith(".weight") or v.dim() != 2:
        return False
    if k.startswith(NVFP4_HIGH_RISK_PREFIXES):
        return False
    return nvfp4_tensorcore_eligible(v)


def should_quantize_mxfp8(k, v):
    if not k.endswith(".weight") or v.dim() != 2:
        return False
    if k.startswith(MXFP8_HIGH_RISK_PREFIXES):
        return False
    return True


def quantize_nvfp4_weight(k, v):
    try:
        from comfy_kitchen.tensor import QuantizedTensor
    except ImportError as e:
        raise SystemExit("nvfp4 precision requires comfy-kitchen") from e

    base = k[:-len(".weight")]
    qt = QuantizedTensor.from_float(v.contiguous(), NVFP4_LAYOUT)
    tensors = qt.state_dict(f"{base}.weight")
    tensors[f"{base}.comfy_quant"] = comfy_quant_tensor("nvfp4")
    return tensors


def quantize_mxfp8_weight(k, v):
    try:
        from comfy_kitchen.tensor import QuantizedTensor
    except ImportError as e:
        raise SystemExit("mxfp8 precision requires comfy-kitchen") from e

    base = k[:-len(".weight")]
    qt = QuantizedTensor.from_float(v.contiguous(), MXFP8_LAYOUT)
    tensors = qt.state_dict(f"{base}.weight")
    scale_key = f"{base}.weight_scale"
    tensors[scale_key] = tensors[scale_key].view(torch.uint8)
    tensors[f"{base}.comfy_quant"] = comfy_quant_tensor("mxfp8")
    return tensors


def int8_convrot_eligible(v):
    # ConvRot rotates along in_features in power-of-4 Hadamard groups; in_features must be a
    # multiple of the group size or comfy_kitchen._rotate_weight raises.
    if v.dim() != 2:
        return False
    return v.shape[1] % INT8_CONVROT_GROUPSIZE == 0


def should_quantize_int8(k, v):
    if not k.endswith(".weight") or v.dim() != 2:
        return False
    if k.startswith(INT8_HIGH_RISK_PREFIXES):
        return False
    return int8_convrot_eligible(v)


def quantize_int8_weight(k, v):
    # Canonical stock-ComfyUI int8_tensorwise producer (comfy-quants), byte-matching
    # ComfyUI's own save path — not comfy-kitchen from_float.
    base = k[:-len(".weight")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qweight, scale, rotated = _quantize_int8_tensorwise_per_row(
        v.contiguous().to(dev), convrot=True, group_size=INT8_CONVROT_GROUPSIZE)
    marker = int8_tensorwise_checkpoint_quant_config(
        convrot=rotated, convrot_groupsize=INT8_CONVROT_GROUPSIZE)
    return {
        f"{base}.weight": qweight.detach().to("cpu").contiguous(),
        f"{base}.weight_scale": scale.detach().to("cpu").contiguous(),
        f"{base}.comfy_quant": torch.tensor(list(json.dumps(marker).encode("utf-8")), dtype=torch.uint8),
    }


def cast(sd, precision):
    out = {}
    nvfp4_quantized = 0
    nvfp4_kept_fp16 = 0
    nvfp4_kept_policy = 0
    nvfp4_kept_shape = 0
    mxfp8_quantized = 0
    mxfp8_kept_fp16 = 0
    mxfp8_kept_policy = 0
    int8_quantized = 0
    int8_kept_fp16 = 0
    int8_kept_policy = 0
    int8_kept_shape = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if precision == "fp16":
            out[k] = v.to(torch.float16)
        elif precision == "fp8_e4m3fn":
            out[k] = v.to(FP8)
        elif precision == "nvfp4":
            if should_quantize_nvfp4(k, v):
                out.update(quantize_nvfp4_weight(k, v))
                nvfp4_quantized += 1
            else:
                out[k] = v.to(torch.float16)
                nvfp4_kept_fp16 += 1
                if k.endswith(".weight") and v.dim() == 2:
                    if not nvfp4_tensorcore_eligible(v):
                        nvfp4_kept_shape += 1
                    elif k.startswith(NVFP4_HIGH_RISK_PREFIXES):
                        nvfp4_kept_policy += 1
        elif precision == "mxfp8":
            if should_quantize_mxfp8(k, v):
                out.update(quantize_mxfp8_weight(k, v))
                mxfp8_quantized += 1
            else:
                out[k] = v.to(torch.float16)
                mxfp8_kept_fp16 += 1
                if k.endswith(".weight") and v.dim() == 2:
                    if k.startswith(MXFP8_HIGH_RISK_PREFIXES):
                        mxfp8_kept_policy += 1
        elif precision == "int8":
            if should_quantize_int8(k, v):
                out.update(quantize_int8_weight(k, v))
                int8_quantized += 1
            else:
                out[k] = v.to(torch.float16)
                int8_kept_fp16 += 1
                if k.endswith(".weight") and v.dim() == 2:
                    if k.startswith(INT8_HIGH_RISK_PREFIXES):
                        int8_kept_policy += 1
                    elif not int8_convrot_eligible(v):
                        int8_kept_shape += 1
        else:
            raise SystemExit(f"unknown precision: {precision}")
    if precision == "nvfp4":
        print(
            f"nvfp4 quantized_weights={nvfp4_quantized} kept_fp16={nvfp4_kept_fp16} "
            f"kept_policy={nvfp4_kept_policy} kept_shape={nvfp4_kept_shape}"
        )
    if precision == "mxfp8":
        print(
            f"mxfp8 quantized_weights={mxfp8_quantized} kept_fp16={mxfp8_kept_fp16} "
            f"kept_policy={mxfp8_kept_policy}"
        )
    if precision == "int8":
        print(
            f"int8 quantized_weights={int8_quantized} kept_fp16={int8_kept_fp16} "
            f"kept_policy={int8_kept_policy} kept_shape={int8_kept_shape} "
            f"convrot_groupsize={INT8_CONVROT_GROUPSIZE}"
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert ByteDance SeedVR2 .pth to ComfyUI safetensors.")
    ap.add_argument("--src", required=True, help="source .pth checkpoint")
    ap.add_argument("--job", action="append", required=True, metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; one source load serves every job")
    ap.add_argument("--cond", default=None, metavar="pos_emb.pt,neg_emb.pt",
                    help="embed text conditioning as positive_conditioning/negative_conditioning")
    ap.add_argument("--dump", action="store_true", help="print source tensor count and dtypes")
    args = ap.parse_args()

    sd = load_state_dict(args.src)

    # A SeedVR2 DiT (it carries the emb_in/txt_in/vid_in input layers a VAE lacks) loads in ComfyUI
    # only with the fixed text conditioning baked in. Refuse to write a DiT safetensors that omits
    # positive_conditioning/negative_conditioning: ComfyUI rejects such a file, so a silent success
    # here would hand back an unusable model. VAE checkpoints have no such layers and need no --cond.
    if not args.cond and any(k.startswith(NVFP4_HIGH_RISK_PREFIXES) for k in sd):
        raise SystemExit(
            "source is a SeedVR2 DiT checkpoint; ComfyUI loads it only with conditioning baked in. "
            "Re-run with --cond pos_emb.pt,neg_emb.pt (a DiT safetensors without "
            "positive_conditioning/negative_conditioning is rejected at load)."
        )

    cond = None
    if args.cond:
        parts = [p.strip() for p in args.cond.split(",")]
        if len(parts) != 2 or not all(parts):
            raise SystemExit("--cond must be 'pos_emb.pt,neg_emb.pt'")
        pos_path, neg_path = parts
        cond = {
            "positive_conditioning": torch.load(pos_path, map_location="cpu", weights_only=True),
            "negative_conditioning": torch.load(neg_path, map_location="cpu", weights_only=True),
        }

    if args.dump:
        tensor_keys = [k for k in sd if torch.is_tensor(sd[k])]
        dtypes = collections.Counter(str(sd[k].dtype) for k in tensor_keys)
        print(f"{len(tensor_keys)} tensors, dtypes={dict(dtypes)}")

    mismatched = []
    for job in args.job:
        precision, sep, remainder = job.partition(":")
        if not sep or not remainder:
            raise SystemExit(f"invalid --job {job!r}; expected PRECISION:OUT[:SHA256]")
        out, expected = remainder, None
        head, sha_sep, tail = remainder.rpartition(":")
        if sha_sep and len(tail) == 64 and all(c in "0123456789abcdefABCDEF" for c in tail):
            out, expected = head, tail.lower()
        tensors = cast(sd, precision)
        if cond:
            tensors.update(cond)
        save_file(tensors, out)
        digest = sha256(out)
        verdict = "" if expected is None else ("  OK" if digest == expected else "  MISMATCH")
        print(f"{precision:30s} {digest}  {out}{verdict}")
        if expected is not None and digest != expected:
            mismatched.append(out)

    if mismatched:
        raise SystemExit(f"SHA256 mismatch: {', '.join(mismatched)}")


if __name__ == "__main__":
    main()
