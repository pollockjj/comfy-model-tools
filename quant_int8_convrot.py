"""INT8 + ConvRot quantizer for comfy-kitchen — auto layer detection (no per-model recipe).

Quantizes the per-token block linears (attention + FFN), passes everything else through.
Recipe: fp32 upcast, block-Hadamard rotation at the per-layer best power-of-4 groupsize,
per-channel absmax scale, embedded `<layer>.comfy_quant` config.

    python quant_int8_auto.py SRC [DST.safetensors] [--dry-run] [options]

SRC: .safetensors (lazy) or torch pickle .pth/.pt/.ckpt (safe weights_only load, held in RAM).
DST optional: defaults to SRC with bf16/fp16/fp32 -> int8_convrot (or _int8_convrot appended).
Auto-detect can't see token count M (small-M/windowed/audio layers get quantized anyway — size
win, maybe not speed) or loader quirks (manual_cast/key-remap loaders -> loads but outputs garbage).
So run --dry-run on an unfamiliar arch and use --min-gemm / --exclude as needed.
"""
# ruff: noqa: T201  (print() is this CLI's output)
import argparse
import json
import os
import re
import time
import collections
import torch
from safetensors import safe_open
from safetensors.torch import save_file
try:
    from comfy_kitchen.tensor.int8 import _build_hadamard, _rotate_weight
except ImportError:
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

VALID_GS  = (256, 64, 16)                       # convrot Hadamard sizes; power-of-4, prefer largest
CLIP_GRID = torch.linspace(0.55, 1.0, 80)
FP8 = (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))

def best_gs(k):
    return next((g for g in VALID_GS if k % g == 0), None)

# safe_open-compatible reader for torch pickle checkpoints (weights_only=True -> safe load, no code
# execution; whole file into RAM since pickle has no lazy access).
_DTYPE_CODE = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32",
               torch.float64: "F64", torch.int8: "I8", torch.uint8: "U8",
               getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
               getattr(torch, "float8_e5m2", None): "F8_E5M2"}

class _TorchSlice:                                  # mimics safetensors get_slice()
    def __init__(self, t): self._t = t
    def get_shape(self): return list(self._t.shape)
    def get_dtype(self): return _DTYPE_CODE.get(self._t.dtype, str(self._t.dtype))

class _TorchReader:
    def __init__(self, path):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        sd = self._find_state_dict(obj)
        if sd is None:
            raise ValueError(f"no tensor state-dict found in {path}")
        self._sd = sd
    @staticmethod
    def _find_state_dict(obj):
        if isinstance(obj, dict):
            tensors = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            if tensors:
                return tensors                       # this level holds the weights
            for key in ("state_dict", "model_state_dict", "model", "module", "net", "ema", "params"):
                if isinstance(obj.get(key), dict):
                    found = _TorchReader._find_state_dict(obj[key])
                    if found:
                        return found
        return None
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def metadata(self): return {}
    def keys(self): return list(self._sd.keys())
    def get_slice(self, k): return _TorchSlice(self._sd[k])
    def get_tensor(self, k): return self._sd[k]

def open_model(path):
    """safe_open for .safetensors; safe (weights_only) torch.load for .pth/.pt/.ckpt/.bin."""
    if path.lower().endswith(".safetensors"):
        return safe_open(path, framework="pt", device="cpu")
    return _TorchReader(path)

# Detection = quantize every eligible 2-D block linear, minus a name denylist. No projection-name
# allowlist (fragile: every arch invents new names like to_qkv/add_q_proj/single_blocks.linear1).
# adaLN MODULATION *is* quantized (it's a big M=batch GEMM that quantizes cleanly ~0.9%, and on
# Qwen/Flux it's 18-33% of the model — no reason to leave it bf16). We only exclude what's genuinely
# not a per-token GEMM worth quantizing: scale_shift buffers, rope/pos_embed, input embedders,
# gate/router logits, M=1 timestep MLPs, output head/final. 1-D norms are dropped by not-2d already.
# Careful bits: `embedder` (not `embed`) keeps `*_embeddings_connector` in; bare `gate` stays
# (SwiGLU) — only gate_logits/router drop; `timestep`/`time` catch the M=1 embed but not the modulator.
EXCLUDE_SEG = re.compile(
    r"scale_shift|rope|rotary|rel_pos|pos_?embed|embedder|"
    r"gate_logits|router|routing|logit|temperature|"
    r"(?:^|_)time|temb|t_emb|guidance|register|refiner_blocks|adapter|"
    r"(?:^|_)(?:final|head|proj_out|out_layer)(?:_|$)")
# `refiner_blocks` = short-M text side-path (Krea txtfusion.refiner_blocks); main-stream refiners are
# `*_refiner` (Boogu/Z-Image), kept. `adapter` = conditioning injection modules (Wan-Animate
# face_adapter, ip/control adapters): tiny, identity/quality-critical, quantize worst -> leave bf16.

def classify(key, shape):
    """Quantize every eligible 2-D block linear except the name denylist. Returns (bool, reason)."""
    if len(shape) != 2:
        return (False, "not-2d")
    n, k = shape
    if n < 8:
        return (False, "small-N")
    gs = best_gs(k)
    if gs is None:
        return (False, "ineligible-K")
    segs = key.split(".")
    # in a block = an integer segment with named structure after it (blocks.5.attn.q). A trailing
    # integer is a Sequential index on a top-level MLP (tmlp.0, img_emb.proj.1) -> not a block.
    if not any(segs[i].isdigit() for i in range(len(segs) - 1)):
        return (False, "not-in-indexed-block")
    if any(EXCLUDE_SEG.search(s) for s in segs):
        return (False, "denylist(scale_shift/embed/gate/time/head/refiner_blocks/adapter)")
    return (True, f"gs{gs}")

# ---------------------------------------------------------------------------
# Quantization (fp32 upcast + block-Hadamard rotation + MSE-optimal per-channel clip)
# ---------------------------------------------------------------------------
@torch.no_grad()
def quantize_convrot(w, gs, mseclip=True, device="cuda"):
    wf = w.to(device, torch.float32)
    h  = _build_hadamard(gs, device=wf.device, dtype=torch.float32)
    wr = _rotate_weight(wf, h, gs)
    absmax = wr.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    if not mseclip:
        scale = (absmax / 127.0).clamp(min=1e-30)
        q = (wr / scale).round().clamp(-127, 127)
        return q.to(torch.int8), scale.to(torch.float32)
    best_mse = torch.full_like(absmax, float("inf"))
    best_scale = absmax / 127.0
    best_q = None
    for a in CLIP_GRID.tolist():
        scale = (absmax * a / 127.0).clamp(min=1e-30)
        q = (wr / scale).round().clamp(-127, 127)
        mse = ((q * scale - wr) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = q.clone() if best_q is None else torch.where(better.expand_as(q), q, best_q)
    return best_q.to(torch.int8), best_scale.to(torch.float32)

@torch.no_grad()
def recon_metrics(qd, scale, w_ref, gs, device="cuda"):
    """Reconstruct (dequant + un-rotate) and return (cosine, relative_error_%)."""
    deq = qd.to(device).float() * scale.to(device)
    h = _build_hadamard(gs, device=device, dtype=torch.float32)
    deq = _rotate_weight(deq, h, gs)
    wf = w_ref.to(device).float()
    cos = torch.nn.functional.cosine_similarity(deq.flatten(), wf.flatten(), dim=0).item()
    relerr = ((deq - wf).norm() / wf.norm().clamp(min=1e-30)).item() * 100.0
    return cos, relerr

@torch.no_grad()
def quantize_convrot_chunked(w, gs, mseclip=True, device="cuda", max_elems=1 << 27):
    """Row-chunked quantize + metrics: identical results (rows are independent), fp32 peak bounded
    by ~3 * max_elems * 4 bytes instead of 3x the full tensor. Metric sums aggregate exactly."""
    rows = max(1, max_elems // max(1, w.shape[1]))
    h = _build_hadamard(gs, device=device, dtype=torch.float32)
    qds, scales = [], []
    dot = deq2 = ref2 = err2 = 0.0
    for i in range(0, w.shape[0], rows):
        wc = w[i:i + rows]
        qd, scale = quantize_convrot(wc, gs, mseclip=mseclip, device=device)
        deq = qd.to(device).float() * scale.to(device)
        deq = _rotate_weight(deq, h, gs)
        wf = wc.to(device, torch.float32)
        dot += (deq * wf).sum().item()
        deq2 += (deq * deq).sum().item()
        ref2 += (wf * wf).sum().item()
        err2 += ((deq - wf) ** 2).sum().item()
        qds.append(qd.cpu())
        scales.append(scale.cpu())
    cos = dot / max((deq2 * ref2) ** 0.5, 1e-30)
    relerr = (err2 / max(ref2, 1e-30)) ** 0.5 * 100.0
    return torch.cat(qds), torch.cat(scales), cos, relerr

def cq_tensor(gs):
    cfg = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
    return torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", help="output .safetensors; if omitted, derived from SRC by "
                    "replacing bf16/fp16/fp32 with int8_convrot (or appending _int8_convrot)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    ap.add_argument("--exclude", default=None, help="regex; matching layers are FORCED to passthrough")
    ap.add_argument("--include", default=None, help="regex; matching eligible layers are FORCED to quantize")
    ap.add_argument("--min-gemm", type=int, default=256,
                    help="skip a layer if min(N,K) < this (default 256: a GEMM whose small side is "
                         "under ~256 never beats bf16 at any M, so int8 is pure overhead). --min-gemm 0 disables.")
    ap.add_argument("--mseclip", action="store_true", help="MSE-optimal clip instead of absmax (~2-3%% lower weight error, but a proxy — validate output before trusting it)")
    ap.add_argument("--downcast-fp32", action="store_true", help="downcast stray fp32 passthrough linears to compute dtype")
    ap.add_argument("--warn-thresh", type=float, default=2.0, help="warn on any quantized layer whose relerr%% exceeds this (default 2.0)")
    ap.add_argument("--verify-report", default=None, help="write the full per-layer (relerr, cos, gs) table to this path")
    ap.add_argument("--device", default="cuda", help="device for the quantize/verify math (default cuda)")
    args = ap.parse_args()
    if not args.dst and not args.dry_run:
        # derive dst from src: swap dtype token for int8_convrot (else append), always .safetensors
        base = os.path.splitext(os.path.basename(args.src))[0]
        new = re.sub(r"(?i)(bf16|fp16|fp32)", "int8_convrot", base)
        if new == base:
            new = base + "_int8_convrot"
        args.dst = os.path.join(os.path.dirname(args.src), new + ".safetensors")
        print(f"auto dst -> {args.dst}")
    exc = re.compile(args.exclude) if args.exclude else None
    inc = re.compile(args.include) if args.include else None

    with open_model(args.src) as st:               # .safetensors (lazy) or .pth/.pt/.ckpt (safe load)
        src_meta = st.metadata() or {}
        keys = list(st.keys())
        scaled = {k[:-len(".weight_scale")] for k in keys if k.endswith(".weight_scale")}  # fp8 sources
        # compute/passthrough dtype = dominant non-fp8 float weight dtype
        dtc = collections.Counter(st.get_slice(k).get_dtype() for k in keys if k.endswith(".weight"))
        target = torch.float16 if dtc.get("F16", 0) >= dtc.get("BF16", 0) and dtc.get("F16", 0) else torch.bfloat16

        # ---- plan ----
        plan = []           # (base, shape, gs)
        skip = collections.Counter()
        for key in keys:
            if not key.endswith(".weight"):
                continue
            base = key[:-len(".weight")]
            shape = tuple(st.get_slice(key).get_shape())
            q, reason = classify(base, shape)
            if exc and exc.search(base):
                q, reason = False, "excluded(flag)"
            if inc and inc.search(base) and len(shape) == 2 and best_gs(shape[1]) and shape[0] >= 8:
                q, reason = True, f"gs{best_gs(shape[1])}(incl-flag)"
            if q and args.min_gemm and min(shape) < args.min_gemm:
                q, reason = False, f"below-min-gemm({min(shape)})"
            if q:
                plan.append((base, shape, best_gs(shape[1])))
            else:
                skip[reason] += 1

        # ---- report ----
        by_pat = collections.defaultdict(lambda: [0, None, None])
        qparams = 0
        for base, shape, gs in plan:
            pat = re.sub(r"\d+", "N", base)
            by_pat[pat][0] += 1
            by_pat[pat][1] = shape
            by_pat[pat][2] = gs
            qparams += shape[0] * shape[1]
        print(f"SRC {args.src}")
        print(f"compute/passthrough dtype: {target}")
        print(f"\nQUANTIZE {len(plan)} layers (int8+convrot, {'MSE-clip' if args.mseclip else 'absmax'}):")
        for pat in sorted(by_pat):
            c, shape, gs = by_pat[pat]
            print(f"  x{c:<4d} gs{gs:<3d} {str(shape):16s} {pat}")
        gsdist = collections.Counter(gs for _, _, gs in plan)
        print(f"  groupsizes: {dict(gsdist)}   quantized params: {qparams/1e9:.2f}B  (~{qparams/1e9:.1f} GB int8)")
        print(f"\nLEAVE AS-IS ({sum(skip.values())} weights):")
        for reason, c in skip.most_common():
            print(f"  x{c:<4d} {reason}")
        if args.dry_run:
            print("\n[dry-run] nothing written.")
            return

        # ---- execute ----
        quant_set = {b for b, _, _ in plan}
        out = {}
        nq = 0
        t0 = time.time()
        errs = []            # (relerr%, cos, gs, base) per quantized layer
        for key in keys:
            if key.endswith(".weight_scale"):
                continue                                  # fp8 source scale: consumed by dequant
            t = st.get_tensor(key)
            if not key.endswith(".weight"):
                out[key] = t
                continue
            base = key[:-len(".weight")]
            # materialize source weight (dequant fp8 rowwise if needed)
            if t.dtype in FP8 and base in scaled:
                sc = st.get_tensor(base + ".weight_scale").float()
                w = t.float() * sc.view(-1, 1)
            elif t.dtype in FP8:
                w = t.float()
            else:
                w = t
            if base in quant_set:
                gs = best_gs(w.shape[1])
                qd, scale, cos, relerr = quantize_convrot_chunked(w, gs, mseclip=args.mseclip, device=args.device)
                assert cos > 0.99, f"BROKEN quant (rotation/format?) {base} cos={cos:.5f} relerr={relerr:.2f}%"
                if relerr > args.warn_thresh:
                    print(f"  WARN high error: {base} gs={gs} relerr={relerr:.2f}% cos={cos:.5f}", flush=True)
                errs.append((relerr, cos, gs, base))
                out[key] = qd.cpu()
                out[f"{base}.weight_scale"] = scale.cpu()
                out[f"{base}.comfy_quant"]  = cq_tensor(gs)
                nq += 1
                if nq % 100 == 0:
                    print(f"  {nq}/{len(plan)} ... {base} gs={gs} relerr={relerr:.2f}% cos={cos:.5f}", flush=True)
            else:
                # passthrough: fp8 must be de-fp8'd; fp32 optionally downcast; else keep source dtype
                if t.dtype in FP8:
                    out[key] = w.to(target)
                elif t.dtype == torch.float32 and args.downcast_fp32 \
                        and not (base.endswith(".scale") or EXCLUDE_SEG.search(base.split(".")[-1])):
                    out[key] = w.to(target)
                else:
                    out[key] = t
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        save_file(out, args.dst, metadata=dict(src_meta))
        print(f"DONE: quantized {nq} layers, {len(out)} tensors, {time.time()-t0:.1f}s -> {args.dst}")

        # ---- per-layer error report ----
        if errs:
            errs.sort(reverse=True)                        # worst relerr first
            rvals = [e[0] for e in errs]
            mean = sum(rvals) / len(rvals)
            over = [e for e in errs if e[0] > args.warn_thresh]
            per_gs = collections.defaultdict(list)
            for r, c, gs, b in errs:
                per_gs[gs].append(r)
            print("\n=== quant error (relerr = ||dequant-source|| / ||source||) ===")
            print(f"  mean {mean:.3f}%   min {min(rvals):.3f}%   max {max(rvals):.3f}%   layers {len(errs)}")
            print("  per groupsize: " + "  ".join(
                f"gs{gs}: mean {sum(v)/len(v):.3f}% max {max(v):.3f}% (x{len(v)})" for gs, v in sorted(per_gs.items())))
            print("  worst 8 layers:")
            for r, c, gs, b in errs[:8]:
                print(f"    {r:6.3f}%  cos {c:.5f}  gs{gs:<3d} {b}")
            if over:
                print(f"  !! {len(over)} layer(s) over --warn-thresh ({args.warn_thresh}%) — review above")
        if args.verify_report and errs:
            with open(args.verify_report, "w") as f:
                f.write("relerr_pct\tcosine\tgroupsize\tlayer\n")
                for r, c, gs, b in errs:
                    f.write(f"{r:.4f}\t{c:.6f}\t{gs}\t{b}\n")
            print(f"  full per-layer table -> {args.verify_report}")

if __name__ == "__main__":
    main()
