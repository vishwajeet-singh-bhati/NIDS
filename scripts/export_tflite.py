"""Turn the ONNX FP32 model (from export_mobile.py) into TFLite files for Android: FP32 and INT8.

    python scripts/export_tflite.py --bundle models/nids_deploy_v1 --dataset data/datasets/full_group_random_s0 \
        --out models/mobile_v1 [--n-calib 512] [--n-eval 3000] [--eval-split test]

Needs (pip):  tensorflow  tf_keras  onnx2tf  ai-edge-litert  onnx_graphsurgeon  sng4onnx  onnxsim  onnxruntime

Writes  <out>/model_fp32.tflite, model_int8.tflite, phone_inputs/*.bin, tflite_report.md

INT8 here = TFLite post-training static quantisation (int8 weights per channel, int8 activations calibrated on TRAIN
windows; the model's inputs/outputs stay float32/int32). Report compares every variant with the ONNX FP32 model on real
windows: logit differences, alert-decision flips at the bundle threshold, window ROC-AUC, size, desktop latency.

TFLite layout note: onnx2tf exposes the inputs channels-first:  meta [1,12,32]  mask [1,32]  payload [1,128,32] int32
agg [1,22]   (i.e. the PyTorch tensors with the last two axes swapped for meta and payload).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_mobile import batches, roc_auc  # noqa: E402
from src.data.normalization import Normalizer  # noqa: E402
from src.training.data import SplitData  # noqa: E402

KINDS = {"meta", "payload", "agg"}


def to_tflite_layout(b):
    return {"meta": b["meta"].transpose(0, 2, 1).astype(np.float32), "mask": b["mask"].astype(np.float32),
            "payload": b["payload"].transpose(0, 2, 1).astype(np.int32), "agg": b["agg"].astype(np.float32)}


def run_tflite(path, rows_batches, threads=1):
    """Returns (logits with NaN for windows that failed, {input name: (shape, dtype)}, [failure diagnostics])."""
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    idx = {}
    for x in it.get_input_details():
        for k in ("meta", "mask", "payload", "agg"):
            if k in x["name"]:
                idx[k] = x["index"]
    out = it.get_output_details()[0]["index"]
    z, fails, n = [], [], 0
    for b in rows_batches:
        t = to_tflite_layout(b)
        for i in range(len(t["mask"])):
            for k in idx:
                it.set_tensor(idx[k], t[k][i:i + 1])
            try:
                it.invoke()
                z.append(float(it.get_tensor(out).reshape(-1)[0]))
            except RuntimeError as e:
                z.append(float("nan"))
                fails.append({"window": n, "error": str(e)[:160], "mask_sum": float(t["mask"][i].sum()),
                              "meta_finite": bool(np.isfinite(t["meta"][i]).all()), "agg_finite": bool(np.isfinite(t["agg"][i]).all()),
                              "mask_values": sorted(set(np.unique(t["mask"][i]).tolist())),
                              "payload_min_max": [int(t["payload"][i].min()), int(t["payload"][i].max())]})
            n += 1
    names = {x["name"]: (x["shape"].tolist(), np.dtype(x["dtype"]).name) for x in it.get_input_details()}
    return np.array(z), names, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, help="folder that already holds model_fp32.onnx (export_mobile.py output)")
    ap.add_argument("--n-calib", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import onnxruntime as ort
    import tensorflow as tf

    out = Path(a.out)
    onnx_fp = out / "model_fp32.onnx"
    if not onnx_fp.is_file():
        raise SystemExit(f"{onnx_fp} missing: run scripts/export_mobile.py first")
    thr = float(json.loads((Path(a.bundle) / "config.json").read_text())["threshold_logit"])
    norm = Normalizer.load(Path(a.dataset) / "normalization.json")
    rng = np.random.default_rng(a.seed)
    tr, ev = SplitData(a.dataset, "train", norm), SplitData(a.dataset, a.eval_split, norm)
    cal_rows = np.sort(rng.choice(len(tr), min(a.n_calib, len(tr)), replace=False))
    ev_rows = np.sort(rng.choice(len(ev), min(a.n_eval, len(ev)), replace=False))
    y = ev.y_all[ev.sel[ev_rows]].astype(int)

    # ---- 1. ONNX -> TFLite float (+ a SavedModel) via onnx2tf
    sm = out / "tf_savedmodel"
    if sm.exists():
        shutil.rmtree(sm)
    tmp_onnx = out / "_b1_copy.onnx"       # onnx2tf rewrites the input model's shapes: never point it at the original
    shutil.copy2(onnx_fp, tmp_onnx)
    cmd = ["onnx2tf", "-i", str(tmp_onnx), "-o", str(sm), "-b", "1", "-fdosm", "-ois", "meta:1,32,12", "mask:1,32",
           "payload:1,32,128", "agg:1,22"]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    tmp_onnx.unlink(missing_ok=True)
    if r.returncode != 0 or not (sm / "saved_model.pb").is_file():
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("onnx2tf failed")
    base = tmp_onnx.stem
    shutil.copy2(sm / f"{base}_float32.tflite", out / "model_fp32.tflite")

    # ---- 2. INT8 (static, calibrated on TRAIN windows)
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))

    def rep():
        for b in batches(tr, cal_rows, KINDS, 1):
            yield to_tflite_layout(b)

    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    (out / "model_int8.tflite").write_bytes(conv.convert())

    # INT8 weights only (dynamic-range): activations stay float, so the mask-sum denominators are never quantised
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    (out / "model_int8_wonly.tflite").write_bytes(conv.convert())

    # ---- 3. reference (ONNX FP32) + every TFLite variant on the same real windows
    sess = ort.InferenceSession(str(onnx_fp), providers=["CPUExecutionProvider"])
    ref = np.concatenate([sess.run(None, {"meta": b["meta"], "mask": b["mask"], "payload": b["payload"].astype(np.int32),
                                          "agg": b["agg"]})[0] for b in batches(ev, ev_rows, KINDS, 256)])
    rep_lines = ["# TFLite export report", "",
                 f"eval windows: {len(ev_rows)} from `{a.eval_split}` | alert threshold (logit) {thr:.3f} | calibration: {len(cal_rows)} TRAIN windows", "",
                 "| variant | size MB | max abs diff vs ONNX FP32 | mean abs diff | decision flips | window ROC-AUC | median ms (1 thread, desktop) |",
                 "|---|---|---|---|---|---|---|",
                 f"| ONNX FP32 (reference) | {onnx_fp.stat().st_size / 1e6:.2f} | - | - | - | {roc_auc(y, ref):.4f} | - |"]
    names_info, fail_lines = {}, []
    for nm in ("model_fp32", "model_int8", "model_int8_wonly"):
        p = out / f"{nm}.tflite"
        z, names, fails = run_tflite(p, batches(ev, ev_rows, KINDS, 256))
        names_info[nm] = names
        ok = np.isfinite(z)
        if fails:
            print(f"!! {nm}: {len(fails)} of {len(z)} windows FAILED to run. First diagnostics:")
            for f in fails[:5]:
                print("   ", f)
            fail_lines.append(f"- `{nm}.tflite`: {len(fails)} of {len(z)} windows failed, e.g. {fails[0]}")
        z, refk, yk = np.where(ok, z, 0.0)[ok], ref[ok], y[ok]
        d = np.abs(z - refk)
        one = next(batches(ev, ev_rows[:1], KINDS, 1))
        t = []
        for _ in range(5):
            t0 = time.perf_counter()
            run_tflite(p, [one])
            t.append((time.perf_counter() - t0) * 1e3)
        rep_lines.append(f"| {nm}.tflite | {p.stat().st_size / 1e6:.2f} | {d.max():.4f} | {d.mean():.5f} | "
                         f"{int(((z >= thr) != (refk >= thr)).sum())} of {len(z)} | {roc_auc(yk, z):.4f} | (see phone benchmark) |")
    if fail_lines:
        rep_lines += ["", "## Windows that FAILED in TFLite (excluded from the rows above)", ""] + fail_lines
    rep_lines += ["", "## Input tensors (names differ per file; use these with the phone benchmark tool)", ""]
    for nm, names in names_info.items():
        rep_lines.append(f"- `{nm}.tflite`: " + "; ".join(f"`{k}` {v[0]} {v[1]}" for k, v in names.items()))

    # ---- 4. raw inputs for ONE real window, for `benchmark_model --input_layer_value_files` on the phone
    pi = out / "phone_inputs"
    pi.mkdir(exist_ok=True)
    one = to_tflite_layout(next(batches(ev, ev_rows[:1], KINDS, 1)))
    for k, v in one.items():
        v.tofile(pi / f"{k}.bin")
    rep_lines += ["", f"`phone_inputs/*.bin`: one real window as raw little-endian arrays; expected logit (ONNX FP32) = {ref[0]:.4f}"]
    (out / "tflite_report.md").write_text("\n".join(rep_lines))
    print("\n".join(rep_lines))


if __name__ == "__main__":
    main()
