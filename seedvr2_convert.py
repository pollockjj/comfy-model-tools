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
  fp8_e4m3fn_mixed_block35_fp16 float8_e4m3fn, but tensors under "blocks.35." kept float16
                                (keeping the last DiT block in fp16 avoids line/tile
                                 artifacts on the 7B model)
  nvfp4                         eligible 2D .weight tensors -> TensorCoreNVFP4Layout; high-risk
                                input/output projections, text/embedding input layers,
                                tensorcore-ineligible shapes, and (7B only) the last block
                                blocks.35 stay float16; everything else -> float16
  mxfp8                         eligible 2D .weight tensors -> TensorCoreMXFP8Layout; high-risk
                                input/output projections, text/embedding input layers,
                                and (7B only) the last block blocks.35 stay float16; everything else -> float16
  int8                          eligible 2D .weight tensors -> TensorWiseINT8Layout with ConvRot
                                (per-channel weight scale + online group-wise Hadamard activation
                                rotation, groupsize 256); high-risk input/output projections,
                                text/embedding input layers, and weights whose in_features is not
                                a multiple of 256 stay float16; everything else -> float16.
                                All DiT blocks are quantized, including the last block (blocks.35
                                on 7B); int8 has no mixed-precision block exemption

Examples:
  # 3B DiT -> fp16 and fp8, conditioning baked in (one load serves both jobs)
  python seedvr2_convert.py --src seedvr2_ema_3b.pth --cond pos_emb.pt,neg_emb.pt \
      --job fp16:seedvr2_3b_fp16.safetensors \
      --job fp8_e4m3fn:seedvr2_3b_fp8_e4m3fn.safetensors

  # 7B DiT -> fp16 and block35-mixed fp8
  python seedvr2_convert.py --src seedvr2_ema_7b.pth --cond pos_emb.pt,neg_emb.pt \
      --job fp16:seedvr2_7b_fp16.safetensors \
      --job fp8_e4m3fn_mixed_block35_fp16:seedvr2_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors

  # VAE (no conditioning)
  python seedvr2_convert.py --src ema_vae.pth --job fp16:ema_vae_fp16.safetensors

  # 3B DiT -> NVFP4 and MXFP8 (ComfyUI-native quantized weights), conditioning baked in
  python seedvr2_convert.py --src seedvr2_ema_3b.pth --cond pos_emb.pt,neg_emb.pt \
      --job nvfp4:seedvr2_3b_nvfp4.safetensors \
      --job mxfp8:seedvr2_3b_mxfp8.safetensors

  # 7B DiT -> NVFP4 and MXFP8 (7B keeps blocks.35 fp16; the 3B above quantizes all blocks)
  python seedvr2_convert.py --src seedvr2_ema_7b.pth --cond pos_emb.pt,neg_emb.pt \
      --job nvfp4:seedvr2_7b_nvfp4_mixed_block35_fp16.safetensors \
      --job mxfp8:seedvr2_7b_mxfp8_mixed_block35_fp16.safetensors

  # 3B DiT -> INT8 ConvRot (ComfyUI-native quantized weights), conditioning baked in
  python seedvr2_convert.py --src seedvr2_ema_3b.pth --cond pos_emb.pt,neg_emb.pt \
      --job int8:seedvr2_3b_int8.safetensors

  # 7B DiT -> INT8 ConvRot (all blocks quantized, including blocks.35)
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
  2742ca6fee63bc5cc1773f426dd4b07b78cad27f51c9ea5cd42b035e6b592252  seedvr2_7b_fp16.safetensors  <- seedvr2_ema_7b.pth, fp16 + cond
  d89ac95ee1566dfc1ee50c6075a2bfe4028d811dd8751f584505de89ef5c4cf3  seedvr2_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors  <- seedvr2_ema_7b.pth, fp8_e4m3fn_mixed_block35_fp16 + cond
  70823bca54b9c24eeb56e1c452697c7c2a430867e58db0e376c6e260f3a4489d  seedvr2_7b_sharp_fp16.safetensors  <- seedvr2_ema_7b_sharp.pth, fp16 + cond
  700ee64fe0859c3df3abfa40c89f3a16068651bf8c8e5294726b6369e7b0d1e3  seedvr2_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors  <- seedvr2_ema_7b_sharp.pth, fp8_e4m3fn_mixed_block35_fp16 + cond
  6acf15dca5bb83556d38b7c06a8e4402a87ef94d0010e974b464855c41eaba6a  seedvr2_3b_nvfp4.safetensors  <- seedvr2_ema_3b.pth, nvfp4 + cond
  cf0d30d90a92424ce77f5836898be05ff5b8e8f731120a02d8642cff5ed4d87c  seedvr2_3b_mxfp8.safetensors  <- seedvr2_ema_3b.pth, mxfp8 + cond
  0ee5d7c4c4aac94fd24e3b68bb6e977aa4c7526e83499e183434cf6cced05fae  seedvr2_7b_nvfp4_mixed_block35_fp16.safetensors  <- seedvr2_ema_7b.pth, nvfp4 + cond
  a9fc925e2a8fd1a1615030e79151c39e0bd6c6f5fab14b454125d94c2497a85f  seedvr2_7b_mxfp8_mixed_block35_fp16.safetensors  <- seedvr2_ema_7b.pth, mxfp8 + cond
  d5814e47c0b8cd968e0477e62a2e2663501c5fd1a2319f6dc8ba03efa35e0d56  seedvr2_7b_sharp_nvfp4_mixed_block35_fp16.safetensors  <- seedvr2_ema_7b_sharp.pth, nvfp4 + cond
  8f8b98662fd6d1919d55b0646a6546b0e57262270264eac4d56f4d352660a4c6  seedvr2_7b_sharp_mxfp8_mixed_block35_fp16.safetensors  <- seedvr2_ema_7b_sharp.pth, mxfp8 + cond
"""
import argparse
import collections
import hashlib
import json

import torch
from safetensors.torch import save_file

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


def mixed_block35_prefix(sd):
    # 3B vs 7B differentiation, matching the fp8 "mixed_block35_fp16" convention and the output
    # filenames (seedvr2_3b_* carry no "mixed_block" suffix; seedvr2_7b_* do): only the 7B keeps
    # its last DiT block (blocks.35) in fp16 to avoid line/tile artifacts. The 3B (last block
    # blocks.31) quantizes every eligible block, exactly like the 3B fp8 checkpoint. Carve the
    # block only when blocks.35 is present (the 7B); return None for the 3B.
    if any(k.startswith("blocks.35.") for k in sd):
        return "blocks.35."
    return None


def nvfp4_tensorcore_eligible(v):
    if v.dim() != 2:
        return False
    return roundup(v.shape[1], 16) % NVFP4_ALIGNMENT == 0


def should_quantize_nvfp4(k, v, last_block_prefix):
    if not k.endswith(".weight") or v.dim() != 2:
        return False
    if last_block_prefix and k.startswith(last_block_prefix):
        return False
    if k.startswith(NVFP4_HIGH_RISK_PREFIXES):
        return False
    return nvfp4_tensorcore_eligible(v)


def should_quantize_mxfp8(k, v, last_block_prefix):
    if not k.endswith(".weight") or v.dim() != 2:
        return False
    if last_block_prefix and k.startswith(last_block_prefix):
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
    try:
        from comfy_kitchen.tensor import QuantizedTensor
    except ImportError as e:
        raise SystemExit("int8 precision requires comfy-kitchen") from e

    base = k[:-len(".weight")]
    # ConvRot weight rotation + rowwise INT8 quant route through comfy_kitchen CUDA kernels.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qt = QuantizedTensor.from_float(
        v.contiguous().to(dev),
        INT8_LAYOUT,
        is_weight=True,
        per_channel=True,
        convrot=True,
        convrot_groupsize=INT8_CONVROT_GROUPSIZE,
    )
    tensors = {key: t.detach().to("cpu").contiguous() for key, t in qt.state_dict(f"{base}.weight").items()}
    tensors[f"{base}.comfy_quant"] = comfy_quant_tensor(
        INT8_FORMAT, {"convrot": True, "convrot_groupsize": INT8_CONVROT_GROUPSIZE}
    )
    return tensors


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
    last_block_prefix = mixed_block35_prefix(sd)
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if precision == "fp16":
            out[k] = v.to(torch.float16)
        elif precision == "fp8_e4m3fn":
            out[k] = v.to(FP8)
        elif precision == "fp8_e4m3fn_mixed_block35_fp16":
            out[k] = v.to(torch.float16) if k.startswith("blocks.35.") else v.to(FP8)
        elif precision == "nvfp4":
            if should_quantize_nvfp4(k, v, last_block_prefix):
                out.update(quantize_nvfp4_weight(k, v))
                nvfp4_quantized += 1
            else:
                out[k] = v.to(torch.float16)
                nvfp4_kept_fp16 += 1
                if k.endswith(".weight") and v.dim() == 2:
                    if not nvfp4_tensorcore_eligible(v):
                        nvfp4_kept_shape += 1
                    elif last_block_prefix and k.startswith(last_block_prefix):
                        nvfp4_kept_policy += 1
                    elif k.startswith(NVFP4_HIGH_RISK_PREFIXES):
                        nvfp4_kept_policy += 1
        elif precision == "mxfp8":
            if should_quantize_mxfp8(k, v, last_block_prefix):
                out.update(quantize_mxfp8_weight(k, v))
                mxfp8_quantized += 1
            else:
                out[k] = v.to(torch.float16)
                mxfp8_kept_fp16 += 1
                if k.endswith(".weight") and v.dim() == 2:
                    if last_block_prefix and k.startswith(last_block_prefix):
                        mxfp8_kept_policy += 1
                    elif k.startswith(MXFP8_HIGH_RISK_PREFIXES):
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
            f"kept_policy={nvfp4_kept_policy} kept_shape={nvfp4_kept_shape} "
            f"last_block={last_block_prefix or 'none'}"
        )
    if precision == "mxfp8":
        print(
            f"mxfp8 quantized_weights={mxfp8_quantized} kept_fp16={mxfp8_kept_fp16} "
            f"kept_policy={mxfp8_kept_policy} last_block={last_block_prefix or 'none'}"
        )
    if precision == "int8":
        print(
            f"int8 quantized_weights={int8_quantized} kept_fp16={int8_kept_fp16} "
            f"kept_policy={int8_kept_policy} kept_shape={int8_kept_shape} "
            f"convrot_groupsize={INT8_CONVROT_GROUPSIZE} last_block=quantized(full-model)"
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
