"""Phase 3: audit how trustworthy the labels are, using only the manifest.

    python -m src.data.label_audit --manifest data/manifests/cic_andmal2017_manifest.csv \
        --out artifacts/reports/label_audit

Writes <out>.md and <out>.json. It does not train anything and changes no labels.

What it measures (all descriptive; none of it proves a label right or wrong)
  1. Counts per label / category / family / benign subset at example, flow and capture level.
  2. "Background share": the fraction of each class that is local/infrastructure traffic
     (responder is a private/link-local/multicast address - e.g. DNS to the gateway). If malware
     captures contain as much of it as benign captures do, those windows carry the capture label
     but no class information => label noise.
  3. Responder overlap: how many remote public IPs are contacted by BOTH classes.
  4. Confound/shortcut checks: capture-date ranges per class, initiator-IP overlap, window-size mix.
  5. Warnings + recommendations.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.dataset_loader import load_manifest
from src.data.flow_tags import add_flow_tags
from src.utils.logging_utils import setup_logging

log = logging.getLogger("label_audit")

USECOLS = ["example_id", "capture_id", "app_id", "malware_family", "malware_category", "capture_date",
           "subset", "original_label", "label", "binary_label", "label_reliability", "flow_id",
           "n_packets", "ip_proto", "initiator_ip", "responder_ip", "responder_port", "n_ts_backward"]


def _share(s: pd.Series) -> Dict[str, float]:
    vc = s.value_counts(normalize=True)
    return {str(k): round(float(v), 4) for k, v in vc.items()}


def build_report(man: pd.DataFrame) -> Dict[str, Any]:
    df = add_flow_tags(man)
    rep: Dict[str, Any] = {"n_examples": int(len(df)), "n_flows": int(df["flow_id"].nunique()),
                           "n_captures": int(df["capture_id"].nunique()), "warnings": [], "classes": {}}
    labels = sorted(df["label"].unique())
    fl = df.drop_duplicates("flow_id")  # one row per flow
    cap = df.groupby("capture_id").agg(label=("label", "first"), n=("example_id", "size"),
                                       app=("app_id", "first"), fam=("malware_family", "first"))
    for lab in labels:
        d, f, c = df[df["label"] == lab], fl[fl["label"] == lab], cap[cap["label"] == lab]
        pub = f[f["responder_scope"] == "public"]
        rep["classes"][lab] = {
            "examples": int(len(d)), "flows": int(len(f)), "captures": int(len(c)),
            "apps": int((d["app_id"] != "").groupby(d["app_id"]).any().sum()) if (d["app_id"] != "").any() else 0,
            "families": int(d.loc[d["malware_family"] != "", "malware_family"].nunique()),
            "windows_per_capture": {k: round(float(v), 1) for k, v in c["n"].describe()[
                ["mean", "min", "50%", "max"]].items()},
            "background_share_examples": round(float((d["flow_tag"] == "local_infra").mean()), 4),
            "background_share_flows": round(float((f["flow_tag"] == "local_infra").mean()), 4),
            "flow_tag_share_examples": _share(d["flow_tag"]),
            "ip_proto_share_examples": _share(d["ip_proto"].astype(str)),
            "short_window_share(<=3pkts)": round(float((d["n_packets"] <= 3).mean()), 4),
            "mean_packets_per_window": round(float(d["n_packets"].mean()), 2),
            "windows_with_ts_backward_share": round(float((d["n_ts_backward"] > 0).mean()), 4)
            if "n_ts_backward" in d else None,
            "public_responder_ips": int(pub["responder_ip"].nunique()),
            "captures_without_app_id": int((c["app"] == "").sum()),
        }
    # ---- composition tables
    mal = df[df["malware_family"] != ""]
    rep["by_category"] = {k: {"captures": int(g["capture_id"].nunique()), "examples": int(len(g))}
                          for k, g in df[df["malware_category"] != ""].groupby("malware_category")}
    rep["by_family"] = {k: {"category": str(g["malware_category"].iloc[0]),
                            "captures": int(g["capture_id"].nunique()), "examples": int(len(g)),
                            "background_share": round(float((g["flow_tag"] == "local_infra").mean()), 3)}
                        for k, g in mal.groupby("malware_family")}
    rep["by_subset"] = {k: {"captures": int(g["capture_id"].nunique()), "examples": int(len(g))}
                        for k, g in df[df["subset"] != ""].groupby("subset")}
    # ---- responder overlap between the two binary classes
    bin_ok = df[df["binary_label"] >= 0]
    sets: Dict[int, set] = {}
    for b in (0, 1):
        if (fl["binary_label"] == b).any():
            sets[b] = set(fl.loc[(fl["binary_label"] == b) & (fl["responder_scope"] == "public"), "responder_ip"])
    if 0 in sets and 1 in sets:
        inter = sets[0] & sets[1]
        union = sets[0] | sets[1]
        pf = fl[fl["responder_scope"] == "public"]
        rep["responder_overlap"] = {
            "benign_public_ips": len(sets[0]), "malicious_public_ips": len(sets[1]),
            "shared_ips": len(inter), "jaccard": round(len(inter) / max(len(union), 1), 4),
            "malicious_flow_share_to_shared_ip": round(float(
                pf.loc[pf["binary_label"] == 1, "responder_ip"].isin(inter).mean()), 4),
            "benign_flow_share_to_shared_ip": round(float(
                pf.loc[pf["binary_label"] == 0, "responder_ip"].isin(inter).mean()), 4),
        }
    # ---- confound checks
    conf: Dict[str, Any] = {}
    dates = df[df["capture_date"] != ""].groupby("label")["capture_date"].agg(["min", "max", "nunique"])
    conf["capture_date_range_by_label"] = {k: {"min": r["min"], "max": r["max"], "distinct_days": int(r["nunique"])}
                                           for k, r in dates.iterrows()}
    if {"benign", "malicious_associated"} <= set(dates.index):
        b, m = dates.loc["benign"], dates.loc["malicious_associated"]
        conf["capture_dates_disjoint"] = bool(b["max"] < m["min"] or m["max"] < b["min"])
    ips = {lab: set(fl.loc[fl["label"] == lab, "initiator_ip"]) for lab in labels}
    conf["initiator_ips_per_label"] = {k: len(v) for k, v in ips.items()}
    if "benign" in ips and "malicious_associated" in ips:
        conf["initiator_ip_overlap"] = len(ips["benign"] & ips["malicious_associated"])
    rep["confounds"] = conf
    # ---- reliability / provenance
    rep["label_reliability"] = {k: int(v) for k, v in df["label_reliability"].value_counts().items()}
    # ---- warnings
    W: List[str] = rep["warnings"]
    cl = rep["classes"]
    if "unknown" in cl:
        W.append(f"{cl['unknown']['examples']} examples are 'unknown' (excluded from supervised work).")
    if "benign" in cl and "malicious_associated" in cl:
        b, m = cl["benign"], cl["malicious_associated"]
        W.append(f"Capture-level imbalance benign:malicious = {b['captures']}:{m['captures']} "
                 f"({b['captures'] / max(m['captures'], 1):.1f}:1); example-level = "
                 f"{b['examples']}:{m['examples']} ({b['examples'] / max(m['examples'], 1):.1f}:1).")
        mb, bb = m["background_share_examples"], b["background_share_examples"]
        if mb >= 0.2:
            W.append(f"LABEL NOISE: {mb:.0%} of malicious_associated windows go to local/infrastructure "
                     f"addresses (benign: {bb:.0%}). Those windows carry a malware label from the capture but "
                     f"are generic chatter.")
        if abs(mb - bb) > 0.15:
            W.append(f"Background share differs a lot between classes ({mb:.0%} vs {bb:.0%}): a model could "
                     f"learn 'amount of background' instead of malware behaviour.")
        ss = (m["short_window_share(<=3pkts)"], b["short_window_share(<=3pkts)"])
        if abs(ss[0] - ss[1]) > 0.15:
            W.append(f"Share of <=3-packet windows differs between classes ({ss[0]:.0%} vs {ss[1]:.0%}).")
        if conf.get("capture_dates_disjoint"):
            W.append("TEMPORAL CONFOUND: benign and malicious captures were recorded on non-overlapping dates "
                     f"({conf['capture_date_range_by_label']}). Network/environment drift between those days "
                     "could be learnable and would NOT generalise.")
        if conf.get("initiator_ip_overlap", 0) == 0 and len(ips.get("benign", ())) and len(ips.get("malicious_associated", ())):
            W.append("Benign and malicious captures use disjoint device IPs: never feed IPs to a model.")
        ro = rep.get("responder_overlap")
        if ro and ro["malicious_flow_share_to_shared_ip"] > 0.3:
            W.append(f"{ro['malicious_flow_share_to_shared_ip']:.0%} of malicious public-IP flows go to servers "
                     f"also contacted by benign captures: those flows cannot be separated by destination.")
    few = [k for k, v in rep["by_family"].items() if v["captures"] < 3]
    if few:
        W.append(f"Families with < 3 captures (hard to split/evaluate by family): {few}")
    no_app = sum(v["captures_without_app_id"] for v in cl.values())
    if no_app:
        W.append(f"{no_app} captures have no app_id (rule fallback): grouping falls back to capture_id.")
    return rep


def to_markdown(rep: Dict[str, Any]) -> str:
    L = ["# Label audit (Phase 3)", "",
         f"examples **{rep['n_examples']}**, flows **{rep['n_flows']}**, captures **{rep['n_captures']}**", "",
         "## Warnings", ""]
    L += [f"- {w}" for w in rep["warnings"]] or ["- none"]
    L += ["", "## Per class", "", "| label | captures | flows | examples | apps | families | windows/capture (mean/min/med/max) | "
          "background share (ex) | <=3-pkt share | mean pkts |", "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for lab, c in rep["classes"].items():
        w = c["windows_per_capture"]
        L.append(f"| {lab} | {c['captures']} | {c['flows']} | {c['examples']} | {c['apps']} | {c['families']} | "
                 f"{w['mean']}/{w['min']}/{w['50%']}/{w['max']} | {c['background_share_examples']:.1%} | "
                 f"{c['short_window_share(<=3pkts)']:.1%} | {c['mean_packets_per_window']} |")
    L += ["", "### Traffic mix per class (share of examples by flow_tag)", ""]
    for lab, c in rep["classes"].items():
        L.append(f"- **{lab}**: " + ", ".join(f"{k} {v:.1%}" for k, v in list(c["flow_tag_share_examples"].items())[:10]))
    L += ["", "### IP protocol mix per class (share of examples)", ""]
    for lab, c in rep["classes"].items():
        L.append(f"- **{lab}**: " + ", ".join(f"proto {k}: {v:.1%}" for k, v in c["ip_proto_share_examples"].items()))
    if rep.get("responder_overlap"):
        L += ["", "## Responder overlap (public IPs, flow level)", "", "```", json.dumps(rep["responder_overlap"], indent=2), "```"]
    L += ["", "## Confound checks", "", "```", json.dumps(rep["confounds"], indent=2), "```"]
    for title, key in (("Malware categories", "by_category"), ("Benign subsets", "by_subset")):
        if rep.get(key):
            L += ["", f"## {title}", "", "| name | captures | examples |", "|---|---:|---:|"]
            L += [f"| {k} | {v['captures']} | {v['examples']} |" for k, v in rep[key].items()]
    if rep["by_family"]:
        L += ["", "## Malware families", "", "| family | category | captures | examples | background share |", "|---|---|---:|---:|---:|"]
        L += [f"| {k} | {v['category']} | {v['captures']} | {v['examples']} | {v['background_share']:.1%} |"
              for k, v in sorted(rep["by_family"].items(), key=lambda kv: (kv[1]["category"], kv[0]))]
    L += ["", "## Label reliability (examples)", "", "```", json.dumps(rep["label_reliability"], indent=2), "```",
          "", "## What this means for the first experiments", "",
          "- Call the task **malware-associated traffic classification with weak capture-level labels**, not intrusion/C&C detection.",
          "- Report every metric twice: all windows, and with `--exclude-local-infra` (see splits.py).",
          "- Evaluate on **held-out families** (`family_holdout` strategy) as well as held-out captures; a random "
          "capture split overestimates generalisation.",
          "- Do not use IPs, ports, dates or capture identifiers as features.",
          "- Better ground truth for flow-level work: correlate flows with the malware's known C&C domains/IPs "
          "(threat-intel lists, sandbox reports, CopperDroid/VirusTotal behaviour reports) and relabel only the "
          "matching flows as `confirmed_attack`; everything else in a malware capture stays `malicious_associated` or `unknown`."]
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="artifacts/reports/label_audit")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    try:
        header = pd.read_csv(a.manifest, nrows=0).columns
        cols = [c for c in USECOLS if c in header]
        man = load_manifest(a.manifest, usecols=cols)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    missing = {"example_id", "capture_id", "label", "flow_id", "responder_ip"} - set(man.columns)
    if missing:
        log.error("Manifest lacks required columns %s - rebuild it with the current build_sequences.", sorted(missing))
        return 2
    rep = build_report(man)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(to_markdown(rep), encoding="utf-8")
    print(f"examples={rep['n_examples']} flows={rep['n_flows']} captures={rep['n_captures']}")
    for lab, c in rep["classes"].items():
        print(f"  {lab:22s} captures={c['captures']:5d} examples={c['examples']:8d} "
              f"background={c['background_share_examples']:.1%} <=3pkt={c['short_window_share(<=3pkts)']:.1%}")
    print("\nWARNINGS:")
    for w in rep["warnings"]:
        print(" -", w)
    print(f"\nReports: {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
