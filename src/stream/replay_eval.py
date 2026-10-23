"""Replay-based evaluation of the REAL-TIME pipeline on held-out captures.

    python -m src.stream.replay_eval --run-dir runs/<ds>/packet_bert_s0 --manifest data/manifests/full.csv \
        --splits data/splits/<name>.csv --raw-root data/raw/CIC-AndMal2017 --out runs/<ds>/packet_bert_s0/stream

Every val/test capture is pushed through `StreamDetector` packet by packet (the same code path as live
capture), recording the score trace of every flow. Because the alert policy is a pure function of a flow's
score trace (`src.stream.policy`), operating points are then swept on the traces without re-running the model.

Protocol (no test peeking):
  1. replay VAL captures -> choose the policy threshold on VAL:   max flow-level recall subject to
     flow-level FPR <= --target-fpr  (default 1 %), or max-F1.
  2. replay TEST captures -> apply that threshold unchanged.

Metrics (all computed, none assumed):
  * flow level   : accuracy, precision, recall, F1, FPR, FNR, confusion matrix, ROC-AUC/PR-AUC of each flow's max score
  * capture level: fraction of benign captures with >= 1 alert, fraction of malware captures with >= 1 alert,
                   capture-level precision/recall/F1/accuracy
  * deployment   : alerts per hour of BENIGN traffic, time-to-detect (seconds and packets since flow start,
                   median/p90) for true-positive flows, throughput (packets/s) and model ms/window
  * breakdowns   : per malware family (recall), per flow tag (local_infra vs rest), results with local
                   infrastructure flows removed.
LABEL CAVEAT: flow and capture labels come from the capture directory (weak). A "false positive" on a
malware capture's benign background chatter is counted as a true positive here, so recall is optimistic and
precision on malware captures is uncertain; benign-capture FPR is the cleanest number.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.flow_tags import add_flow_tags
from src.stream.engine import StreamConfig, StreamDetector
from src.stream.policy import PolicyParams, first_alert
from src.utils import metrics as M
from src.utils.logging_utils import setup_logging

log = logging.getLogger("replay")


# --------------------------------------------------------------------------------------- capture table
def capture_table(manifest: str, splits: str, raw_root: Optional[str], split_names: Sequence[str]) -> pd.DataFrame:
    man = pd.read_csv(manifest, low_memory=False)
    sp = pd.read_csv(splits)
    sp = sp[sp["split"].isin(split_names)]
    j = man.merge(sp[["example_id", "split"]], on="example_id", how="inner")
    cols = [c for c in ("capture_id", "capture_path", "binary_label", "malware_family", "malware_category", "subset", "split")
            if c in j.columns]
    caps = j[cols].groupby("capture_id", as_index=False).first()
    if raw_root:
        caps["path"] = [str(Path(raw_root) / p) for p in caps["capture_path"]]
    else:
        caps["path"] = caps["capture_path"]
    return caps[caps["binary_label"].isin([0, 1])].reset_index(drop=True)


# --------------------------------------------------------------------------------------- replay
def replay_captures(det: StreamDetector, caps: pd.DataFrame, max_packets: int = 0) -> Dict[str, pd.DataFrame]:
    """-> {"flows": one row per flow with its logit trace, "captures": per capture duration/throughput}."""
    flow_rows: List[Dict[str, Any]] = []
    cap_rows: List[Dict[str, Any]] = []
    det.record_scores = True
    for i, c in caps.iterrows():
        det.reset_capture()
        det.scores = []
        t0 = time.perf_counter()
        pk0 = det.stats["packets"]
        first_ts = last_ts = None
        # neutralise alerting during replay: traces are what we keep (policy is applied afterwards)
        saved = det.sc.threshold
        det.sc.threshold = 1e9
        from src.data.parse_pcap import iter_packets
        n_seen = 0
        for pkt in iter_packets(c["path"], det.cfg.parse, det.parse_stats):
            if first_ts is None:
                first_ts = pkt.ts
            last_ts = pkt.ts
            n_seen += 1
            det.feed(pkt)
        for f in det.tracker.drain_finished():
            det._finalise(f)  # noqa: SLF001
        for f in det.tracker.flush():
            det._finalise(f)  # noqa: SLF001
        det.run_inference()
        det.sc.threshold = saved
        wall = time.perf_counter() - t0
        tr = pd.DataFrame(det.scores)
        cap_rows.append({"capture_id": c["capture_id"], "packets": n_seen, "duration_s": (last_ts - first_ts) if first_ts else 0.0,
                         "wall_s": wall, "n_flows_scored": int(tr["flow_index"].nunique()) if len(tr) else 0})
        if len(tr):
            for fi, g in tr.groupby("flow_index"):
                g = g.sort_values(["ts", "start", "k"], kind="stable")
                r0 = g.iloc[0]
                m = {"initiator_ip": r0["initiator_ip"], "responder_ip": r0["responder_ip"],
                     "responder_port": int(r0["responder_port"]), "ip_proto": int(r0["ip_proto"]),
                     "flow_start_ts": float(r0["flow_start_ts"])}
                flow_rows.append({"capture_id": c["capture_id"], "flow_index": int(fi), "y": int(c["binary_label"]),
                                  "malware_family": c.get("malware_family"), "split": c["split"],
                                  "logits": g["logit"].to_numpy(), "ts": g["ts"].to_numpy(), "k": g["k"].to_numpy(),
                                  "npk": g["n_packets"].to_numpy(), **m})
        if (i + 1) % 25 == 0:
            log.info("replayed %d/%d captures", i + 1, len(caps))
    flows = pd.DataFrame(flow_rows)
    if len(flows):
        flows = add_flow_tags(flows)
    return {"flows": flows, "captures": pd.DataFrame(cap_rows)}


# --------------------------------------------------------------------------------------- policy sweep
def apply_policy(flows: pd.DataFrame, p: PolicyParams) -> pd.DataFrame:
    idx, why, t_det, k_det, pk_det = [], [], [], [], []
    for r in flows.itertuples():
        i, reason = first_alert(r.logits, p)
        idx.append(i)
        why.append(reason)
        if i is None:
            t_det.append(np.nan)
            k_det.append(np.nan)
            pk_det.append(np.nan)
        else:
            t_det.append(float(r.ts[i] - r.flow_start_ts))
            k_det.append(int(r.k[i]))
            pk_det.append(int(r.npk[i]))
    out = flows.copy()
    out["alert"] = [i is not None for i in idx]
    out["alert_reason"] = why
    out["delay_s"] = t_det
    out["decided_on_k"] = k_det
    out["packets_at_alert"] = pk_det
    return out


def flow_confusion(df: pd.DataFrame) -> Dict[str, Any]:
    y, a = df["y"].to_numpy().astype(int), df["alert"].to_numpy().astype(int)
    tp, fp = int(((y == 1) & (a == 1)).sum()), int(((y == 0) & (a == 1)).sum())
    fn, tn = int(((y == 1) & (a == 0)).sum()), int(((y == 0) & (a == 0)).sum())
    d = lambda x, z: float(x / z) if z else float("nan")  # noqa: E731
    prec, rec = d(tp, tp + fp), d(tp, tp + fn)
    return {"n": int(len(df)), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": d(tp + tn, len(df)),
            "precision": prec, "recall": rec, "f1": d(2 * prec * rec, prec + rec) if prec == prec and rec == rec else float("nan"),
            "fpr": d(fp, fp + tn), "fnr": d(fn, fn + tp), "balanced_accuracy": float(np.nanmean([rec, d(tn, tn + fp)]))}


def choose_threshold(val_flows: pd.DataFrame, params: PolicyParams, target_fpr: float, objective: str) -> Dict[str, Any]:
    allz = np.concatenate([z for z in val_flows["logits"]]) if len(val_flows) else np.zeros(1)
    cand = np.unique(np.quantile(allz, np.linspace(0.0, 1.0, 400)))
    cand = np.concatenate([cand, [cand.max() + 1.0]])
    best, rows = None, []
    for t in cand:
        r = flow_confusion(apply_policy(val_flows, PolicyParams(float(t), params.ema_alpha, params.min_hits, params.hard_margin)))
        rows.append({"threshold": float(t), **{k: r[k] for k in ("recall", "fpr", "precision", "f1")}})
        ok = r["fpr"] <= target_fpr if objective == "fpr" else True
        key = (r["recall"] if objective == "fpr" else (r["f1"] if r["f1"] == r["f1"] else -1), -r["fpr"])
        if ok and r["recall"] == r["recall"] and (best is None or key > best[0]):
            best = (key, float(t), r)
    if best is None:                                    # nothing satisfies the constraint -> most conservative
        t = float(cand[-1])
        return {"threshold": t, "val": flow_confusion(apply_policy(val_flows, PolicyParams(t, params.ema_alpha, params.min_hits, params.hard_margin))),
                "note": "no threshold met the target FPR on val; using the most conservative one"}
    return {"threshold": best[1], "val": best[2], "sweep": rows}


# --------------------------------------------------------------------------------------- report
def capture_level(df: pd.DataFrame, caps: pd.DataFrame, cap_stats: pd.DataFrame) -> Dict[str, Any]:
    per = df.groupby("capture_id").agg(alerts=("alert", "sum"), flows=("alert", "size")).reset_index()
    t = caps[["capture_id", "binary_label", "malware_family"]].merge(per, on="capture_id", how="left").fillna({"alerts": 0, "flows": 0})
    t = t.merge(cap_stats[["capture_id", "duration_s", "packets"]], on="capture_id", how="left")
    y, a = t["binary_label"].to_numpy().astype(int), (t["alerts"].to_numpy() > 0).astype(int)
    tp, fp = int(((y == 1) & (a == 1)).sum()), int(((y == 0) & (a == 1)).sum())
    fn, tn = int(((y == 1) & (a == 0)).sum()), int(((y == 0) & (a == 0)).sum())
    d = lambda x, z: float(x / z) if z else float("nan")  # noqa: E731
    prec, rec = d(tp, tp + fp), d(tp, tp + fn)
    ben = t[t["binary_label"] == 0]
    hrs = float(ben["duration_s"].sum()) / 3600.0
    return {"n_captures": int(len(t)), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": d(tp + tn, len(t)),
            "precision": prec, "recall": rec, "f1": d(2 * prec * rec, prec + rec) if prec == prec and rec == rec else float("nan"),
            "benign_capture_false_alarm_rate": d(fp, fp + tn), "benign_hours": hrs,
            "alerts_per_benign_hour": float(ben["alerts"].sum() / hrs) if hrs > 0 else float("nan"),
            "per_family_capture_recall": {str(f): {"n": int(len(g)), "detected": int((g["alerts"] > 0).sum()),
                                                  "recall": float((g["alerts"] > 0).mean())}
                                          for f, g in t[t["binary_label"] == 1].groupby("malware_family")}}


def split_report(flows: pd.DataFrame, caps: pd.DataFrame, cap_stats: pd.DataFrame, p: PolicyParams) -> Dict[str, Any]:
    df = apply_policy(flows, p) if len(flows) else flows
    rep: Dict[str, Any] = {"policy": vars(p)}
    if not len(df):
        return {**rep, "error": "no scored flows"}
    rep["flow_level"] = flow_confusion(df)
    mx = np.array([z.max() for z in df["logits"]])
    rep["flow_level"]["roc_auc_of_max_window_logit"] = M.safe_roc_auc(df["y"].to_numpy().astype(int), mx)
    rep["flow_level"]["pr_auc_of_max_window_logit"] = M.safe_pr_auc(df["y"].to_numpy().astype(int), mx)
    keep = df["flow_tag"] != "local_infra"
    rep["flow_level_excluding_local_infra"] = flow_confusion(df[keep]) if keep.any() else {}
    rep["by_flow_tag"] = {t: {k: v for k, v in flow_confusion(g).items() if k in ("n", "recall", "fpr", "precision")}
                          for t, g in df.groupby("flow_tag")}
    tp = df[(df["y"] == 1) & df["alert"]]
    if len(tp):
        rep["time_to_detect_true_positive_flows"] = {
            "median_s": float(tp["delay_s"].median()), "p90_s": float(tp["delay_s"].quantile(0.9)),
            "median_packets": float(tp["packets_at_alert"].median()), "p90_packets": float(tp["packets_at_alert"].quantile(0.9)),
            "decided_on_prefix": {int(k): int(v) for k, v in tp["decided_on_k"].value_counts().sort_index().items()}}
    rep["per_family_flow_recall"] = {str(f): float(g["alert"].mean()) for f, g in df[df["y"] == 1].groupby("malware_family")}
    rep["capture_level"] = capture_level(df, caps, cap_stats)
    pk, wall = float(cap_stats["packets"].sum()), float(cap_stats["wall_s"].sum())
    rep["throughput"] = {"packets": int(pk), "wall_s": wall, "packets_per_s": pk / wall if wall else float("nan"),
                         "note": "includes pure-Python packet parsing + flow tracking + model, single process"}
    return rep


def run(run_dir: str, manifest: str, splits: str, raw_root: Optional[str], out_dir: str, target_fpr: float = 0.01,
        objective: str = "fpr", sc: Optional[StreamConfig] = None, device: str = "cpu", max_captures: int = 0,
        capture_stride: int = 1, test_split: str = "test", data_config: str = "configs/data.yaml") -> Dict[str, Any]:
    sc = sc or StreamConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = StreamDetector(str(Path(run_dir) / "best.pt"), stream_cfg=sc, device=device, config_path=data_config)
    results: Dict[str, Any] = {"run": str(run_dir), "model": det.ck["model_name"], "stream_config": vars(sc),
                               "label_caveat": "weak capture-level labels; see src/stream/replay_eval.py docstring"}
    replay: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sp in ("val", test_split):
        caps = capture_table(manifest, splits, raw_root, [sp])
        if capture_stride > 1:
            caps = caps.iloc[::capture_stride].reset_index(drop=True)
        if max_captures:
            caps = caps.iloc[:max_captures]
        log.info("%s: replaying %d captures (benign=%d malware=%d)", sp, len(caps), int((caps.binary_label == 0).sum()),
                 int((caps.binary_label == 1).sum()))
        replay[sp] = {**replay_captures(det, caps), "caps": caps}
        pd.to_pickle(replay[sp]["flows"], out / f"traces_{sp}.pkl")
    base = PolicyParams(0.0, sc.ema_alpha, sc.min_hits, sc.hard_margin)
    if not len(replay["val"]["flows"]) or not len(replay[test_split]["flows"]):
        raise ValueError("no flows were scored in val or test; check --raw-root / capture paths")
    ch = choose_threshold(replay["val"]["flows"], base, target_fpr, objective)
    p = PolicyParams(ch["threshold"], base.ema_alpha, base.min_hits, base.hard_margin)
    results["threshold_selection"] = {"objective": objective, "target_fpr": target_fpr, "threshold_logit": p.threshold,
                                      "val_at_threshold": ch["val"], "note": ch.get("note", "")}
    for sp in ("val", test_split):
        results[sp] = split_report(replay[sp]["flows"], replay[sp]["caps"], replay[sp]["captures"], p)
    (out / "stream_metrics.json").write_text(json.dumps(results, indent=2, default=lambda o: None if o != o else str(o)))
    (out / "stream_report.md").write_text(render_md(results, test_split))
    return results


def _f(x, nd=3):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def render_md(res: Dict[str, Any], test_split: str) -> str:
    L = [f"# Stream replay report - {res['model']}", "", f"Run: `{res['run']}`", "",
         f"Threshold (val-selected, {res['threshold_selection']['objective']}, target FPR "
         f"{res['threshold_selection']['target_fpr']}): logit {res['threshold_selection']['threshold_logit']:.3f}", "",
         f"> {res['label_caveat']}", ""]
    for sp in ("val", test_split):
        r = res[sp]
        if "flow_level" not in r:
            continue
        fl, cl = r["flow_level"], r["capture_level"]
        L += [f"## {sp}", "", "| level | n | accuracy | precision | recall | F1 | FPR |", "|---|---|---|---|---|---|---|",
              f"| flow | {fl['n']} | {_f(fl['accuracy'])} | {_f(fl['precision'])} | {_f(fl['recall'])} | {_f(fl['f1'])} | {_f(fl['fpr'])} |"]
        x = r.get("flow_level_excluding_local_infra") or {}
        if x:
            L.append(f"| flow, no local-infra | {x['n']} | {_f(x['accuracy'])} | {_f(x['precision'])} | {_f(x['recall'])} | {_f(x['f1'])} | {_f(x['fpr'])} |")
        L.append(f"| capture | {cl['n_captures']} | {_f(cl['accuracy'])} | {_f(cl['precision'])} | {_f(cl['recall'])} | {_f(cl['f1'])} | {_f(cl['benign_capture_false_alarm_rate'])} |")
        L += ["", f"Flow ROC-AUC (max window logit) {_f(fl['roc_auc_of_max_window_logit'])}, PR-AUC {_f(fl['pr_auc_of_max_window_logit'])}; "
              f"alerts per benign hour {_f(cl['alerts_per_benign_hour'], 1)} over {_f(cl['benign_hours'], 2)} h."]
        t = r.get("time_to_detect_true_positive_flows")
        if t:
            L.append(f"Time to detect (TP flows): median {_f(t['median_s'])} s / {_f(t['median_packets'], 0)} packets, "
                     f"p90 {_f(t['p90_s'])} s / {_f(t['p90_packets'], 0)} packets. Decided on prefix: {t['decided_on_prefix']}")
        th = r["throughput"]
        L += [f"Throughput {_f(th['packets_per_s'], 0)} packets/s ({th['note']}).", ""]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="training run dir containing best.pt")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True, help="splits CSV used to build the consolidated dataset")
    ap.add_argument("--raw-root", default=None, help="prefix for capture_path in the manifest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--objective", choices=["fpr", "f1"], default="fpr")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--ema-alpha", type=float, default=0.5)
    ap.add_argument("--hard-margin", type=float, default=3.0)
    ap.add_argument("--max-captures", type=int, default=0)
    ap.add_argument("--capture-stride", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    sc = StreamConfig(checkpoints=tuple(a.checkpoints), min_hits=a.min_hits, ema_alpha=a.ema_alpha, hard_margin=a.hard_margin)
    try:
        res = run(a.run_dir, a.manifest, a.splits, a.raw_root, a.out or str(Path(a.run_dir) / "stream"), a.target_fpr,
                  a.objective, sc, a.device, a.max_captures, a.capture_stride, data_config=a.data_config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    print((Path(a.out or Path(a.run_dir) / "stream") / "stream_report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
