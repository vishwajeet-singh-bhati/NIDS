"""Dataset audit: walk a raw dataset directory and report what is actually there.

Usage:
    python -m src.data.inspect_dataset --root data/raw/CIC-AndMal2017 \
        --out artifacts/reports/audit_cic_andmal2017 [--probe-packets 2000] [--count-packets]

Outputs <out>.json and <out>.md. NOTHING is discarded or relabeled here: problems are listed.

What is reported
  * file inventory by type (pcap/pcapng, csv, json, archives, other) with counts and sizes
  * every capture file: format, link type, probe packet count, first/last timestamp in the probe,
    zero-size / unreadable / truncated / unsupported-link-type findings
  * every CSV: columns, dtypes, row count (up to --csv-max-rows), missing values per column,
    sample rows, and CANDIDATE label columns with their value counts
  * every JSON: top-level structure summary
  * "path vocabulary": directory-name tokens and their frequency - use it to write label rules
  * label-granularity HINTS (clearly marked as hints, derived only from observable structure)

The hints are heuristics about structure, not documented facts. Verify them against the
dataset's own documentation before relying on them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.parse_pcap import ParseStats, iter_packets
from src.utils.config import ParseConfig
from src.utils.logging_utils import setup_logging

log = logging.getLogger("inspect_dataset")

PCAP_EXT = {".pcap", ".pcapng", ".cap"}
CSV_EXT = {".csv", ".tsv"}
JSON_EXT = {".json", ".jsonl"}
ARCHIVE_EXT = {".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".tar"}
APK_EXT = {".apk"}
TEXT_EXT = {".txt", ".md", ".log", ".yaml", ".yml", ".xml", ".pdf"}

LABEL_NAME_HINTS = ("label", "class", "category", "family", "malware", "type", "benign", "attack")
FLOW_COLUMN_HINTS = ("src", "dst", "source", "destination", "port", "flow id", "flow_id", "protocol")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in PCAP_EXT:
        return "pcap"
    if ext in CSV_EXT:
        return "csv"
    if ext in JSON_EXT:
        return "json"
    if ext in ARCHIVE_EXT or path.name.lower().endswith(".tar.gz"):
        return "archive"
    if ext in APK_EXT:
        return "apk"
    if ext in TEXT_EXT:
        return "text"
    return "other"


def inspect_pcap(path: Path, probe_packets: int, count_all: bool) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size, "findings": []}
    if info["size_bytes"] == 0:
        info["findings"].append("zero_size_file")
        return info
    cfg = ParseConfig(payload_bytes_kept=0, max_packets_per_capture=0 if count_all else probe_packets)
    stats = ParseStats()
    n = 0
    for _ in iter_packets(path, cfg, stats):
        n += 1
    info.update({k: v for k, v in stats.as_dict().items() if k != "path"})
    info["packets_parsed_in_probe"] = n
    info["probe_was_complete"] = bool(count_all and not stats.file_truncated_or_corrupt) or (
        not stats.hit_packet_limit)
    if stats.open_error:
        info["findings"].append(f"unreadable: {stats.open_error}")
    if stats.file_truncated_or_corrupt:
        info["findings"].append("file_truncated_or_corrupt")
    if stats.unsupported_linktype:
        info["findings"].append(f"unsupported_linktype({stats.linktype})")
    if stats.total_records == 0 and not stats.open_error:
        info["findings"].append("no_packet_records")
    if stats.total_records and stats.non_ip == stats.total_records:
        info["findings"].append("no_ip_packets_in_probe")
    if stats.malformed:
        info["findings"].append(f"malformed_packets={stats.malformed}")
    if stats.ip_fragment_nonfirst_skipped:
        info["findings"].append(f"ip_fragments_nonfirst={stats.ip_fragment_nonfirst_skipped}")
    if stats.truncated_by_snaplen:
        info["findings"].append(f"snaplen_truncated={stats.truncated_by_snaplen}")
    if stats.first_ts and stats.last_ts:
        info["probe_span_s"] = stats.last_ts - stats.first_ts
    return info


def inspect_csv(path: Path, max_rows: int, sample_rows: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size, "findings": []}
    if info["size_bytes"] == 0:
        info["findings"].append("zero_size_file")
        return info
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, nrows=max_rows, sep=sep, low_memory=False, encoding_errors="replace")
    except Exception as exc:
        info["findings"].append(f"unreadable: {type(exc).__name__}: {exc}")
        return info
    info["rows_read"] = int(len(df))
    info["rows_capped"] = bool(len(df) >= max_rows)
    info["n_columns"] = int(df.shape[1])
    info["columns"] = [str(c) for c in df.columns]
    info["dtypes"] = {str(c): str(t) for c, t in df.dtypes.items()}
    miss = df.isna().sum()
    info["missing_values"] = {str(c): int(v) for c, v in miss.items() if v > 0}
    num = df.select_dtypes("number")
    if not num.empty:
        inf_cols = {str(c): int(((num[c] == float("inf")) | (num[c] == float("-inf"))).sum())
                    for c in num.columns}
        info["infinite_values"] = {c: v for c, v in inf_cols.items() if v > 0}
    info["sample_rows"] = json.loads(df.head(sample_rows).to_json(orient="records", default_handler=str))
    cand = {}
    for c in df.columns:
        name = str(c).strip().lower()
        if any(h in name for h in LABEL_NAME_HINTS) or (df[c].dtype == object and df[c].nunique() <= 30):
            vc = df[c].value_counts(dropna=False).head(30)
            cand[str(c)] = {str(k): int(v) for k, v in vc.items()}
    info["candidate_label_columns"] = cand
    lower = [str(c).strip().lower() for c in df.columns]
    flow_like = [c for c in lower if any(h in c for h in FLOW_COLUMN_HINTS)]
    info["granularity_hint"] = (
        "rows look flow-level (columns mention address/port/flow id): " + ", ".join(flow_like[:8])
        if len(flow_like) >= 3 else "no obvious per-flow identifier columns (rows may be per-sample/app)")
    dup = int(df.duplicated().sum())
    if dup:
        info["findings"].append(f"duplicate_rows_in_read_portion={dup}")
    return info


def inspect_json(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size, "findings": []}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            if path.suffix.lower() == ".jsonl":
                first = [json.loads(line) for _, line in zip(range(3), fh) if line.strip()]
                info["structure"] = f"jsonl, first records keys: {[list(x)[:15] if isinstance(x, dict) else type(x).__name__ for x in first]}"
            else:
                obj = json.load(fh)
                if isinstance(obj, dict):
                    info["structure"] = f"object with keys: {list(obj)[:25]}"
                elif isinstance(obj, list):
                    info["structure"] = f"list of {len(obj)}; first item type {type(obj[0]).__name__ if obj else 'n/a'}"
                else:
                    info["structure"] = type(obj).__name__
    except Exception as exc:
        info["findings"].append(f"unreadable: {type(exc).__name__}: {exc}")
    return info


def path_vocabulary(root: Path, files: List[Path]) -> Dict[str, Any]:
    by_depth: Dict[int, Counter] = defaultdict(Counter)
    for f in files:
        parts = f.relative_to(root).parts[:-1]
        for d, name in enumerate(parts):
            by_depth[d][name] += 1
    return {f"depth_{d}": dict(c.most_common(60)) for d, c in sorted(by_depth.items())}


def audit(root: Path, probe_packets: int, count_all: bool, csv_max_rows: int, sample_rows: int) -> Dict[str, Any]:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    by_type: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        by_type[classify(f)].append(f)
    report: Dict[str, Any] = {
        "root": str(root), "total_files": len(files),
        "total_size_bytes": int(sum(f.stat().st_size for f in files)),
        "inventory": {t: {"count": len(v), "size_bytes": int(sum(x.stat().st_size for x in v))}
                      for t, v in sorted(by_type.items())},
        "extensions": dict(Counter(f.suffix.lower() or "(none)" for f in files).most_common()),
        "path_vocabulary": path_vocabulary(root, files),
        "pcaps": [], "csvs": [], "jsons": [], "problems": [],
    }
    for i, f in enumerate(by_type.get("pcap", []), 1):
        rel = f.relative_to(root).as_posix()
        log.info("[pcap %d/%d] %s (%s)", i, len(by_type["pcap"]), rel, human(f.stat().st_size))
        try:
            info = inspect_pcap(f, probe_packets, count_all)
        except Exception as exc:  # defensive: an audit must survive a bad file
            info = {"path": str(f), "findings": [f"inspection_crash: {type(exc).__name__}: {exc}"]}
        info["rel_path"] = rel
        report["pcaps"].append(info)
        for fnd in info.get("findings", []):
            report["problems"].append({"file": rel, "problem": fnd})
    for f in by_type.get("csv", []):
        rel = f.relative_to(root).as_posix()
        log.info("[csv] %s", rel)
        info = inspect_csv(f, csv_max_rows, sample_rows)
        info["rel_path"] = rel
        report["csvs"].append(info)
        for fnd in info.get("findings", []):
            report["problems"].append({"file": rel, "problem": fnd})
    for f in by_type.get("json", []):
        rel = f.relative_to(root).as_posix()
        info = inspect_json(f)
        info["rel_path"] = rel
        report["jsons"].append(info)
        for fnd in info.get("findings", []):
            report["problems"].append({"file": rel, "problem": fnd})
    for f in by_type.get("archive", []):
        report["problems"].append({"file": f.relative_to(root).as_posix(),
                                   "problem": "archive_not_extracted (extract it; contents are NOT inspected)"})
    # structural hints only
    hints = []
    if by_type.get("pcap"):
        hints.append("PCAP files exist: any label derived from their directory/file names is a "
                     "CAPTURE-LEVEL (weak) label, not a per-flow or per-packet label.")
    for c in report["csvs"]:
        if c.get("granularity_hint", "").startswith("rows look flow-level"):
            hints.append(f"{c['rel_path']}: {c['granularity_hint']}")
    if report["csvs"] and by_type.get("pcap"):
        hints.append("Both PCAPs and CSVs are present: check the dataset documentation for how CSV rows "
                     "map to PCAP flows before using CSV labels as flow labels.")
    report["label_granularity_hints"] = hints
    return report


def to_markdown(r: Dict[str, Any]) -> str:
    L: List[str] = [f"# Dataset audit: `{r['root']}`", "",
                    f"- files: **{r['total_files']}**, total size **{human(r['total_size_bytes'])}**", ""]
    L += ["## Inventory", "", "| type | files | size |", "|---|---:|---:|"]
    for t, v in r["inventory"].items():
        L.append(f"| {t} | {v['count']} | {human(v['size_bytes'])} |")
    L += ["", "## Path vocabulary (use this to write rules in configs/labels.yaml)", ""]
    for depth, vocab in r["path_vocabulary"].items():
        L.append(f"**{depth}**: " + ", ".join(f"`{k}` ({v})" for k, v in list(vocab.items())[:40]))
        L.append("")
    L += ["## Problems", ""]
    L += [f"- `{p['file']}`: {p['problem']}" for p in r["problems"]] or ["- none found"]
    L += ["", "## Label-granularity hints (heuristic, verify against dataset documentation)", ""]
    L += [f"- {h}" for h in r["label_granularity_hints"]] or ["- none"]
    if r["pcaps"]:
        L += ["", "## Captures", "", "| file | size | format | linktype | records(probe) | parsed IP pkts | span s | findings |",
              "|---|---:|---|---:|---:|---:|---:|---|"]
        for p in r["pcaps"][:500]:
            L.append(f"| {p['rel_path']} | {human(p.get('size_bytes', 0))} | {p.get('file_format', '')} | "
                     f"{p.get('linktype', '')} | {p.get('total_records', '')} | {p.get('parsed', '')} | "
                     f"{p.get('probe_span_s', ''):.1f} | {'; '.join(p.get('findings', []))} |"
                     if isinstance(p.get('probe_span_s'), float) else
                     f"| {p['rel_path']} | {human(p.get('size_bytes', 0))} | {p.get('file_format', '')} | "
                     f"{p.get('linktype', '')} | {p.get('total_records', '')} | {p.get('parsed', '')} |  | "
                     f"{'; '.join(p.get('findings', []))} |")
        if len(r["pcaps"]) > 500:
            L.append(f"\n(first 500 of {len(r['pcaps'])} shown; full list in the JSON report)")
    for c in r["csvs"]:
        L += ["", f"## CSV `{c['rel_path']}`", ""]
        if "columns" not in c:
            L.append(f"unreadable/empty: {c.get('findings')}")
            continue
        L += [f"- rows read: {c['rows_read']}{' (capped)' if c['rows_capped'] else ''}, columns: {c['n_columns']}",
              f"- granularity hint: {c['granularity_hint']}",
              f"- columns: {', '.join('`'+x+'`' for x in c['columns'][:80])}",
              f"- columns with missing values: {c['missing_values'] or 'none'}"]
        if c.get("infinite_values"):
            L.append(f"- columns with +/-inf: {c['infinite_values']}")
        for col, vc in c["candidate_label_columns"].items():
            L.append(f"- candidate label column `{col}`: {vc}")
    for j in r["jsons"]:
        L += ["", f"## JSON `{j['rel_path']}`", "", f"- {j.get('structure', j.get('findings'))}"]
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset directory to audit")
    ap.add_argument("--out", default="artifacts/reports/dataset_audit", help="output path prefix")
    ap.add_argument("--probe-packets", type=int, default=2000,
                    help="packets read per capture for validation (fast). Ignored with --count-packets")
    ap.add_argument("--count-packets", action="store_true", help="read every packet of every capture (slow)")
    ap.add_argument("--csv-max-rows", type=int, default=200000)
    ap.add_argument("--sample-rows", type=int, default=3)
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    root = Path(a.root)
    if not root.is_dir():
        log.error("Dataset root does not exist or is not a directory: %s", root)
        log.error("Download the dataset manually from its official page (see README) and extract it there.")
        return 2
    rep = audit(root, a.probe_packets, a.count_packets, a.csv_max_rows, a.sample_rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(to_markdown(rep), encoding="utf-8")
    inv = ", ".join(f"{t}={v['count']}" for t, v in rep["inventory"].items())
    print(f"\nAudit complete: {rep['total_files']} files ({inv}); {len(rep['problems'])} problem(s) flagged.")
    print(f"Reports: {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    for p in rep["problems"][:15]:
        print(f"  ! {p['file']}: {p['problem']}")
    if len(rep["problems"]) > 15:
        print(f"  ... {len(rep['problems']) - 15} more in the report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
