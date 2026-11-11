"""Export the deployed detector for phone testing: FP32 ONNX and INT8 ONNX (static, QDQ), with parity checks.

    python scripts/export_mobile.py --bundle models/nids_deploy_v1 --dataset data/datasets/full_group_random_s0 \
        --out models/mobile_v1 [--n-calib 512] [--n-eval 3000] [--eval-split test]

Writes  <out>/model_fp32.onnx, model_int8.onnx, sample_windows.npz (model inputs + expected logits, for the phone
        parity test), export_report.md.

What is compared (all on REAL windows from --eval-split; INT8 calibration uses windows from the TRAIN split only):
  torch QAT-simulated (what was trained/evaluated)  vs  ONNX FP32  vs  ONNX INT8
  -> max/mean |logit difference|, alert-decision flips at the bundle threshold, ROC-AUC against the window labels,
     file sizes, and CPU latency (batch 1) on THIS machine (a phone needs its own measurement).

The exported graph takes the already-normalised tensors (what `tensors_from_arrays` returns). The Android app must
reproduce the parsing / feature extraction / normalisation (src/data, normalization.json) to feed it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.normalization import Normalizer
from src.models.model_factory import load_checkpoint
from src.training.data import SplitData


class Wrapped(torch.nn.Module):
    """Positional-input wrapper (ONNX wants tensors, not a dict). payload is int32 on the wire, cast inside."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, meta, mask, payload, agg):
        return self.m({"meta": meta, "mask": mask, "payload": payload.long(), "agg": agg})


def batches(data: SplitData, rows: np.ndarray, kinds, bs: int):
    for i in range(0, len(rows), bs):
        b = data.batch(rows[i:i + bs], kinds)
        yield {k: v.numpy() for k, v in b.items()}


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = Path(a.bundle)
    cfg = json.loads((bundle / "config.json").read_text())
    thr = float(cfg["threshold_logit"])
    model, ck = load_checkpoint(bundle / "model.pt", "cpu")
    model.eval()
    kinds = set(model.inputs)
    assert kinds == {"meta", "mask", "payload", "agg"} - {"mask"} or kinds >= {"meta", "payload", "agg"}, kinds
    norm = Normalizer.load(Path(a.dataset) / "normalization.json")
    rng = np.random.default_rng(a.seed)

    tr = SplitData(a.dataset, "train", norm)
    ev = SplitData(a.dataset, a.eval_split, norm)
    cal_rows = np.sort(rng.choice(len(tr), min(a.n_calib, len(tr)), replace=False))
    ev_rows = np.sort(rng.choice(len(ev), min(a.n_eval, len(ev)), replace=False))
    y_ev = ev.y_all[ev.sel[ev_rows]].astype(int)

    # ---------- reference logits: torch with QAT simulation ON (as trained) and OFF (plain FP32 of same weights)
    def torch_logits(qat_on: bool):
        model.enable_qat(qat_on)
        res = []
        with torch.no_grad():
            for b in batches(ev, ev_rows, kinds, 256):
                res.append(model({k: torch.from_numpy(v) for k, v in b.items()}).numpy())
        return np.concatenate(res)

    qat_active_in_ckpt = bool(model.qat_active_buf.item()) if hasattr(model, "qat_active_buf") else False
    z_qat = torch_logits(True)
    z_fp = torch_logits(False)

    # ---------- ONNX FP32 export (QAT simulation off: fake-quant is a training device, INT8 is made by the quantiser)
    model.enable_qat(False)
    ex = next(batches(ev, ev_rows[:2], kinds, 2))
    args = (torch.from_numpy(ex["meta"]), torch.from_numpy(ex["mask"]), torch.from_numpy(ex["payload"]).int(),
            torch.from_numpy(ex["agg"]))
    names_in = ["meta", "mask", "payload", "agg"]
    fp32 = out / "model_fp32.onnx"
    torch.onnx.export(Wrapped(model).eval(), args, str(fp32), input_names=names_in, output_names=["logit"],
                      dynamic_axes={n: {0: "batch"} for n in names_in + ["logit"]}, opset_version=a.opset, dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp32)))

    def ort_run(path, rows, bs=256, threads=1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        res = []
        for b in batches(ev, rows, kinds, bs):
            res.append(sess.run(None, {"meta": b["meta"], "mask": b["mask"], "payload": b["payload"].astype(np.int32),
                                       "agg": b["agg"]})[0])
        return np.concatenate(res), sess

    z_onnx_fp, _ = ort_run(fp32, ev_rows)

    # ---------- INT8 (static QDQ, per-channel weights, calibration on TRAIN windows)
    pre = out / "model_fp32_pre.onnx"
    try:
        quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
        src = pre
    except Exception as e:  # pre-processing is optional
        print("quant_pre_process skipped:", e)
        src = fp32

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(batches(tr, cal_rows, kinds, 32))

        def get_next(self):
            b = next(self.it, None)
            if b is None:
                return None
            return {"meta": b["meta"], "mask": b["mask"], "payload": b["payload"].astype(np.int32), "agg": b["agg"]}

    int8 = out / "model_int8.onnx"
    quantize_static(str(src), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    if pre.exists():
        pre.unlink()
    z_onnx_i8, _ = ort_run(int8, ev_rows)

    # ---------- sample windows for the phone parity test (inputs + the expected logits from each variant)
    k = min(256, len(ev_rows))
    sb = next(batches(ev, ev_rows[:k], kinds, k))
    np.savez_compressed(out / "sample_windows.npz", meta=sb["meta"], mask=sb["mask"], payload=sb["payload"].astype(np.int32),
                        agg=sb["agg"], y=y_ev[:k], logit_torch_qat=z_qat[:k], logit_fp32=z_onnx_fp[:k], logit_int8=z_onnx_i8[:k],
                        threshold=np.array([thr], dtype=np.float32))

    # ---------- latency on this machine, batch 1
    ex1 = next(batches(ev, ev_rows[:1], kinds, 1))
    feed = {"meta": ex1["meta"], "mask": ex1["mask"], "payload": ex1["payload"].astype(np.int32), "agg": ex1["agg"]}

    def lat(path, threads):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        for _ in range(20):
            s.run(None, feed)
        t = []
        for _ in range(200):
            t0 = time.perf_counter()
            s.run(None, feed)
            t.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(t)), float(np.percentile(t, 95))

    def cmp(name, z, ref, ref_name):
        d = np.abs(z - ref)
        flips = int(((z >= thr) != (ref >= thr)).sum())
        return f"| {name} vs {ref_name} | {d.max():.4f} | {d.mean():.5f} | {flips} of {len(z)} |"

    sz = lambda p: p.stat().st_size / 1e6
    rep = ["# Mobile export report", "",
           f"bundle: `{bundle}` | eval windows: {len(ev_rows)} from `{a.eval_split}` | alert threshold (logit): {thr:.3f} | "
           f"QAT active in checkpoint: {qat_active_in_ckpt}", "",
           "## Agreement (logit differences; flips = alert decision changes at the bundle threshold)", "",
           "| comparison | max abs diff | mean abs diff | decision flips |", "|---|---|---|---|",
           cmp("ONNX FP32", z_onnx_fp, z_fp, "torch FP32 (QAT off)"),
           cmp("ONNX INT8", z_onnx_i8, z_qat, "torch QAT-simulated"),
           cmp("ONNX INT8", z_onnx_i8, z_onnx_fp, "ONNX FP32"),
           cmp("torch FP32 (QAT off)", z_fp, z_qat, "torch QAT-simulated"), "",
           "## Window-level ROC-AUC against the window labels (weak, capture-level labels)", "",
           "| variant | ROC-AUC |", "|---|---|",
           f"| torch QAT-simulated | {roc_auc(y_ev, z_qat):.4f} |", f"| torch FP32 (QAT off) | {roc_auc(y_ev, z_fp):.4f} |",
           f"| ONNX FP32 | {roc_auc(y_ev, z_onnx_fp):.4f} |", f"| ONNX INT8 | {roc_auc(y_ev, z_onnx_i8):.4f} |", "",
           "## Size and CPU latency on this machine (batch 1; NOT a phone measurement)", "",
           "| file | size MB | median ms (1 thread) | p95 ms (1 thread) |", "|---|---|---|---|"]
    for p in (fp32, int8):
        m, p95 = lat(p, 1)
        rep.append(f"| {p.name} | {sz(p):.2f} | {m:.2f} | {p95:.2f} |")
    rep += ["", "Phone parity test: feed `sample_windows.npz` to the phone and compare to `logit_fp32` / `logit_int8` inside it."]
    (out / "export_report.md").write_text("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    main()
