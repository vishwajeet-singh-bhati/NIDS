"""PCAP -> flows -> fixed-size windows -> NPZ shards + manifest.

Usage (from repo root):
    python -m src.data.build_sequences --config configs/data.yaml
    python -m src.data.build_sequences --config configs/data.yaml --dry-run-labels
    python -m src.data.build_sequences --config configs/data.yaml --window-size 16 --stride 16 \
        --processed-root data/processed/w16 --manifest data/manifests/w16.csv

One NPZ shard is written per capture file: <processed_root>/<capture_id>.npz. Re-running skips
captures whose shard already exists unless --overwrite is given. A per-capture parse/flow audit
is appended to <manifest dir>/processing_log.jsonl. Nothing is silently discarded: every skipped
flow/packet class is counted in that log and in the manifest `warnings` column.

Window policy (documented, deterministic)
-----------------------------------------
For a flow with n kept packets and window W, stride S:
  * starts = 0, S, 2S, ... while start < n; a window beyond the first is emitted only if at least
    `min_tail_packets` packets remain at/after `start` (so tiny leftover tails are not turned
    into near-empty examples) - counted as `windows_dropped_tail`.
  * at most `max_windows_per_flow` windows per flow (rest counted `windows_dropped_cap`).
  * flows (or the first window) with fewer than `min_packets` real packets are skipped and
    counted as `flows_skipped_short`.
  * Short flows are padded to W with an explicit mask.
All windows of one flow share `flow_id`; Phase 4 splitting groups on capture/flow so overlapping
windows can never straddle splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.feature_extraction import count_backward_steps, encode_window
from src.data.flow_tracker import Flow, FlowTracker
from src.data.labeling import LabelMapper
from src.data.parse_pcap import ParseStats, iter_packets
from src.utils.config import META_CHANNELS, PipelineConfig, load_config
from src.utils.logging_utils import setup_logging

log = logging.getLogger("build_sequences")

MANIFEST_COLUMNS = [
    "example_id", "source_dataset", "capture_id", "capture_path", "app_id", "malware_family",
    "malware_category", "capture_date", "subset", "original_label", "label", "binary_label", "label_source",
    "label_reliability", "label_rule", "flow_id", "window_index", "window_start_packet",
    "n_packets", "n_ts_backward", "flow_packets_kept", "flow_total_packets_seen", "window_first_ts",
    "window_last_ts", "window_duration_s", "initiator_ip", "initiator_port", "responder_ip",
    "responder_port", "ip_proto", "flow_end_reason", "shard_file", "shard_row", "warnings",
    "feature_spec_version",
]


def capture_id_for(rel_path: str) -> str:
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:8]
    stem = Path(rel_path).with_suffix("").as_posix().replace("/", "__")
    stem = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)[:80]
    return f"{stem}-{h}"


def find_captures(raw_root: Path, extensions: List[str]) -> List[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def window_starts(n: int, cfg: PipelineConfig) -> Tuple[List[int], int, int]:
    """Return (starts, dropped_tail, dropped_cap) for a flow with n kept packets."""
    W, S = cfg.sequence.window_size, cfg.sequence.stride
    starts: List[int] = []
    dropped_tail = dropped_cap = 0
    start = 0
    while start < n:
        if start > 0 and (n - start) < cfg.sequence.min_tail_packets:
            dropped_tail += 1
            break
        if len(starts) >= cfg.sequence.max_windows_per_flow:
            dropped_cap += 1
            break
        starts.append(start)
        if start + W >= n:  # this window already reaches the last packet
            break
        start += S
    return starts, dropped_tail, dropped_cap


def plan_windows(flows: List[Flow], cfg: PipelineConfig, counters: Dict[str, int]
                 ) -> List[Tuple[Flow, int, int]]:
    """List every (flow, window_index, start_packet) that WOULD be emitted, without encoding."""
    plan: List[Tuple[Flow, int, int]] = []
    for flow in flows:
        n = len(flow.packets)
        if n < cfg.sequence.min_packets:
            counters["flows_skipped_short"] = counters.get("flows_skipped_short", 0) + 1
            continue
        starts, d_tail, d_cap = window_starts(n, cfg)
        counters["windows_dropped_tail"] = counters.get("windows_dropped_tail", 0) + d_tail
        counters["windows_dropped_cap"] = counters.get("windows_dropped_cap", 0) + d_cap
        for wi, s0 in enumerate(starts):
            if len(flow.packets[s0:s0 + cfg.sequence.window_size]) < cfg.sequence.min_packets:
                counters["windows_dropped_short"] = counters.get("windows_dropped_short", 0) + 1
                continue
            plan.append((flow, wi, s0))
    return plan


def sample_plan(plan: List[Tuple[Flow, int, int]], cfg: PipelineConfig, capture_id: str,
                counters: Dict[str, int]) -> List[Tuple[Flow, int, int]]:
    """Deterministic uniform sample of at most `max_examples_per_capture` windows (order kept).
    Sampling depends only on (capture_id, sampling_seed, plan size) - never on labels or features."""
    cap = cfg.sequence.max_examples_per_capture
    counters["windows_dropped_sampling"] = 0
    if not cap or len(plan) <= cap:
        return plan
    seed = int(hashlib.sha1(f"{capture_id}|{cfg.sequence.sampling_seed}".encode()).hexdigest()[:8], 16)
    keep = np.sort(np.random.default_rng(seed).choice(len(plan), size=cap, replace=False))
    counters["windows_dropped_sampling"] = len(plan) - cap
    return [plan[i] for i in keep]


def build_examples_for_flows(flows: List[Flow], cfg: PipelineConfig, capture_id: str,
                             counters: Dict[str, int]) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    """Encode all (or a deterministic sample of) windows of one capture."""
    W = cfg.sequence.window_size
    B = cfg.parse.payload_bytes_kept
    plan = sample_plan(plan_windows(flows, cfg, counters), cfg, capture_id, counters)
    examples: List[Dict[str, np.ndarray]] = []
    rows: List[Dict[str, Any]] = []
    for flow, wi, s0 in plan:
        n = len(flow.packets)
        pk = flow.packets[s0:s0 + W]
        flow_id = f"{capture_id}:f{flow.flow_index}"
        examples.append(encode_window(pk, W, B, cfg.sequence.include_payload, cfg.sequence.iat_log_unit_ms))
        warns = list(flow.warnings)
        if len(pk) < W:
            warns.append("padded_short_window")
        if flow.retransmissions:
            warns.append(f"retx_dropped={flow.retransmissions}" if cfg.flow.drop_tcp_retransmissions
                         else f"retx_kept={flow.retransmissions}")
        if flow.packets_dropped_cap:
            warns.append(f"flow_packets_over_cap={flow.packets_dropped_cap}")
        if any("snaplen_truncated" in p.warnings for p in pk):
            warns.append("snaplen_truncated_payload")
        n_back = count_backward_steps(pk)
        if n_back:
            warns.append("ts_backward_in_window")
        rows.append({
            "example_id": f"{flow_id}:w{wi}", "flow_id": flow_id, "window_index": wi,
            "window_start_packet": s0, "n_packets": len(pk), "n_ts_backward": n_back,
            "flow_packets_kept": n, "flow_total_packets_seen": flow.total_packets_seen,
            "window_first_ts": pk[0].ts, "window_last_ts": pk[-1].ts,
            "window_duration_s": max(0.0, max(p.ts for p in pk) - pk[0].ts),
            "initiator_ip": flow.initiator[0], "initiator_port": flow.initiator[1],
            "responder_ip": flow.responder[0], "responder_port": flow.responder[1],
            "ip_proto": flow.ip_proto, "flow_end_reason": flow.end_reason,
            "warnings": ";".join(sorted(set(warns))),
        })
    return examples, rows


def process_capture(args: Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]], str, bool]
                    ) -> Dict[str, Any]:
    """Worker: process one capture. Takes/returns plain data so it works with multiprocessing."""
    pcap_path, rel_path, cfg_dict, label_cfg, processed_root, overwrite = args
    from src.utils.config import (DatasetConfig, FlowConfig, ParseConfig, PipelineConfig as PC,
                                  SequenceConfig)
    cfg = PC(feature_spec_version=cfg_dict["feature_spec_version"],
             dataset=DatasetConfig(**cfg_dict["dataset"]), parse=ParseConfig(**cfg_dict["parse"]),
             flow=FlowConfig(**cfg_dict["flow"]), sequence=SequenceConfig(**cfg_dict["sequence"]))
    cap_id = capture_id_for(rel_path)
    shard_name = f"{cap_id}.npz"
    shard_path = Path(processed_root) / shard_name
    result: Dict[str, Any] = {"capture_path": rel_path, "capture_id": cap_id, "shard_file": shard_name,
                              "status": "ok", "rows": [], "error": ""}
    t0 = time.time()
    try:
        mapper = LabelMapper(label_cfg or {}, None, warn=False) if label_cfg is not None else None
        label = mapper.label_capture(rel_path) if mapper else None
        stats = ParseStats()
        tracker = FlowTracker(cfg.flow)
        flows = tracker.run(iter_packets(pcap_path, cfg.parse, stats))
        counters: Dict[str, int] = {}
        examples, rows = build_examples_for_flows(flows, cfg, cap_id, counters)
        result["parse_stats"] = stats.as_dict()
        result["flow_stats"] = {**tracker.stats, "flows": len(flows), **counters}
        if stats.open_error:
            result["status"] = "open_error"
            result["error"] = stats.open_error
            return result
        if not examples:
            result["status"] = "no_examples"
            return result
        cap_warns = stats.warning_tags()
        for i, r in enumerate(rows):
            r.update({
                "source_dataset": cfg.dataset.name, "capture_id": cap_id, "capture_path": rel_path,
                "app_id": label.app_id if label else "", "malware_family": label.malware_family if label else "",
                "malware_category": label.category if label else "",
                "capture_date": label.capture_date if label else "",
                "subset": label.subset if label else "",
                "original_label": label.original_label if label else "",
                "label": label.mapped_label if label else "unknown",
                "binary_label": label.binary_label if label else -1,
                "label_source": label.label_source if label else "no_label_config",
                "label_reliability": label.reliability if label else "none",
                "label_rule": label.rule_name if label else "none",
                "shard_file": shard_name, "shard_row": i,
                "feature_spec_version": cfg.feature_spec_version,
            })
            if cap_warns:
                r["warnings"] = ";".join(filter(None, [r["warnings"], *cap_warns]))
        result["rows"] = rows
        shard_exists = shard_path.exists()
        if shard_exists and not overwrite:
            result["status"] = "shard_exists_manifest_rebuilt"
            # still verify row count to avoid a silently inconsistent manifest
            with np.load(shard_path) as z:
                if z["meta"].shape[0] != len(rows):
                    result["status"] = "shard_row_mismatch"
                    result["error"] = (f"existing shard has {z['meta'].shape[0]} rows but pipeline "
                                       f"produced {len(rows)}; rerun with --overwrite")
                    result["rows"] = []
            return result
        Path(processed_root).mkdir(parents=True, exist_ok=True)
        arrays = {k: np.stack([e[k] for e in examples]) for k in examples[0]}
        arrays["feature_spec_json"] = np.array(json.dumps(cfg.feature_spec(), sort_keys=True))
        arrays["meta_channels"] = np.array(META_CHANNELS)
        arrays["example_ids"] = np.array([r["example_id"] for r in rows])
        tmp = shard_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(shard_path)
    except Exception as exc:  # keep the batch alive; the failure is recorded, never swallowed
        log.exception("Failed processing %s", rel_path)
        result["status"] = "exception"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["rows"] = []
    result["seconds"] = round(time.time() - t0, 3)
    return result


def run(cfg: PipelineConfig, label_cfg_path: Optional[str], workers: int, overwrite: bool,
        limit: int, dry_run_labels: bool, capture_stride: int = 1) -> int:
    raw_root = Path(cfg.dataset.raw_root)
    if not raw_root.is_dir():
        log.error("raw_root does not exist: %s (see README for the expected layout)", raw_root)
        return 2
    captures = find_captures(raw_root, cfg.dataset.pcap_extensions)
    if not captures:
        log.error("No capture files with extensions %s under %s", cfg.dataset.pcap_extensions, raw_root)
        return 2
    if capture_stride > 1:  # every K-th capture of the sorted list: spans ALL classes/families, unlike --limit
        captures = captures[::capture_stride]
        log.info("--capture-stride %d -> %d captures", capture_stride, len(captures))
    if limit:
        captures = captures[:limit]
    import yaml
    lc_path = Path(label_cfg_path or cfg.dataset.label_config)
    if not lc_path.is_file():
        log.error("Label config not found: %s", lc_path)
        return 2
    with open(lc_path, "r", encoding="utf-8") as fh:
        label_cfg = yaml.safe_load(fh) or {}
    mapper = LabelMapper(label_cfg)
    rels = [p.relative_to(raw_root).as_posix() for p in captures]

    if dry_run_labels:
        counts: Dict[str, int] = {}
        for rel in rels:
            r = mapper.label_capture(rel)
            counts[f"{r.mapped_label} (rule={r.rule_name})"] = counts.get(
                f"{r.mapped_label} (rule={r.rule_name})", 0) + 1
            print(f"{r.mapped_label:22s} rule={r.rule_name:28s} fam={r.malware_family or '-':20s} "
                  f"app={r.app_id or '-':20s} {rel}")
        print("\nSummary:")
        for k, v in sorted(counts.items()):
            print(f"  {v:6d}  {k}")
        return 0

    cfg_dict = json.loads(cfg.to_json())
    jobs = [(str(p), rel, cfg_dict, label_cfg, cfg.dataset.processed_root, overwrite)
            for p, rel in zip(captures, rels)]
    log.info("Processing %d captures with %d worker(s); window=%d stride=%d payload=%s(%dB)",
             len(jobs), workers, cfg.sequence.window_size, cfg.sequence.stride,
             cfg.sequence.include_payload, cfg.parse.payload_bytes_kept)

    manifest_path = Path(cfg.dataset.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = manifest_path.parent / "processing_log.jsonl"
    all_rows: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as logfh:
        it = (map(process_capture, jobs) if workers <= 1
              else ProcessPoolExecutor(max_workers=workers).map(process_capture, jobs))
        for i, res in enumerate(it, 1):
            all_rows.extend(res["rows"])
            status_counts[res["status"]] = status_counts.get(res["status"], 0) + 1
            rec = {k: v for k, v in res.items() if k != "rows"}
            rec["n_examples"] = len(res["rows"])
            logfh.write(json.dumps(rec, default=str) + "\n")
            if res["status"] not in ("ok", "shard_exists_manifest_rebuilt"):
                log.warning("[%d/%d] %s -> %s %s", i, len(jobs), res["capture_path"], res["status"],
                            res.get("error", ""))
            elif i % 10 == 0 or i == len(jobs):
                log.info("[%d/%d] %d examples so far (%.1fs)", i, len(jobs), len(all_rows), time.time() - t0)
    if not all_rows:
        log.error("No examples were produced. Status counts: %s. See %s", status_counts, log_path)
        return 1
    df = pd.DataFrame(all_rows).reindex(columns=MANIFEST_COLUMNS)
    df = df.sort_values(["capture_id", "example_id"], kind="stable").reset_index(drop=True)
    df.to_csv(manifest_path, index=False)
    log.info("Wrote manifest %s (%d rows). Capture status counts: %s", manifest_path, len(df), status_counts)
    log.info("Label distribution: %s", df["label"].value_counts().to_dict())
    if (df["label"] == "unknown").all():
        log.warning("ALL examples are 'unknown' - no label rule matched. Fix configs/labels.yaml "
                    "(use --dry-run-labels) before any supervised training.")
    bad = {k: v for k, v in status_counts.items() if k not in ("ok", "shard_exists_manifest_rebuilt")}
    return 0 if not bad else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--label-config", default=None, help="override dataset.label_config")
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--processed-root", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--window-size", type=int, default=None, choices=[16, 32, 64])
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--payload-bytes", type=int, default=None)
    ap.add_argument("--no-payload", action="store_true", help="do not write payload arrays at all")
    ap.add_argument("--max-packets-per-capture", type=int, default=None)
    ap.add_argument("--max-examples-per-capture", type=int, default=None,
                    help="deterministic uniform sample of windows per capture (0/None = keep all). "
                         "Needed for the full AndMal2017 set to fit in RAM/disk")
    ap.add_argument("--sampling-seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only process the first N captures (sorted; biased to the first class)")
    ap.add_argument("--capture-stride", type=int, default=1,
                    help="process every K-th capture of the sorted list (deterministic, spans all classes)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run-labels", action="store_true",
                    help="only show which label rule matches each capture; process nothing")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    overrides = {
        "dataset.raw_root": a.raw_root, "dataset.processed_root": a.processed_root,
        "dataset.manifest_path": a.manifest, "sequence.window_size": a.window_size,
        "sequence.stride": a.stride, "parse.payload_bytes_kept": a.payload_bytes,
        "parse.max_packets_per_capture": a.max_packets_per_capture,
        "sequence.max_examples_per_capture": a.max_examples_per_capture,
        "sequence.sampling_seed": a.sampling_seed,
    }
    if a.no_payload:
        overrides["sequence.include_payload"] = False
    if a.window_size and a.stride is None:
        overrides["sequence.stride"] = a.window_size  # keep non-overlapping unless told otherwise
    try:
        cfg = load_config(a.config, overrides)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Config error: %s", exc)
        return 2
    return run(cfg, a.label_config, a.workers, a.overwrite, a.limit, a.dry_run_labels, a.capture_stride)


if __name__ == "__main__":
    sys.exit(main())
