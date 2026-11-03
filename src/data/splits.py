"""Phase 4: leakage-safe train/val/test splits.

    python -m src.data.splits --manifest data/manifests/cic_andmal2017_manifest.csv \
        --name group_random_s0 --strategy group_random --seed 0
    python -m src.data.splits --manifest ... --name family_holdout_s0 --strategy family_holdout --seed 0 \
        --test-families Koler,Dowgin --exclude-local-infra

Output (data/splits/<name>.{csv,json,md}): one row per manifest example with its split.

How leakage is prevented
------------------------
* The unit of assignment is a GROUP, never a window, flow or packet. A group is a connected
  component of captures linked by a shared app_id (e.g. two captures of the same package) and,
  for `family_holdout`, by a shared malware family. So all windows of a flow, all flows of a
  capture, all captures of an app (and, for family_holdout, all captures of a family) land in
  exactly one split. Overlapping/sliding windows therefore can never straddle splits.
* Assignment is stratified by class: each class is split separately toward the requested
  fractions (measured in EXAMPLES, not groups), using a seeded shuffle of the sorted groups and a
  greedy "largest remaining deficit" rule. Only manifest metadata is used - never tensors.
* Filters (`unknown` labels, optional `--exclude-local-infra`, `--min-window-packets`) are rule
  based, label-independent (except `unknown`) and applied identically to every split BEFORE
  assignment. Excluded rows get split = "excluded" and a reason.
* After assignment, `check_leakage` verifies that no group, capture, flow, app (and family for
  family_holdout) occurs in more than one of train/val/test. The CLI exits 1 if any check fails.
* Normalisation statistics are NOT computed here; they are fitted on the train split only
  (src/data/normalization.py) after consolidation.

Balancing classes is NOT done here (no duplication/oversampling before the split). Phase 5 handles
imbalance with loss weights / samplers inside the training process only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.dataset_loader import load_manifest
from src.data.flow_tags import add_flow_tags
from src.utils.logging_utils import setup_logging

log = logging.getLogger("splits")

SPLITS = ("train", "val", "test")
USECOLS = ["example_id", "capture_id", "app_id", "malware_family", "malware_category", "capture_date", "subset",
           "label", "binary_label", "flow_id", "n_packets", "ip_proto", "responder_ip", "responder_port"]
STRATEGIES = ("group_random", "family_holdout")


# ----------------------------------------------------------------------------------- grouping
class _UF:
    def __init__(self):
        self.p: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root: the lexicographically smaller one
            self.p[max(ra, rb)] = min(ra, rb)


def build_groups(df: pd.DataFrame, link_families: bool) -> pd.Series:
    """Group id per row. Captures are linked through app_id and (optionally) malware family.
    Family links apply only to malicious rows (benign rows have no family)."""
    caps = df.drop_duplicates("capture_id")[["capture_id", "app_id", "malware_family"]]
    uf = _UF()
    for cid, app, fam in caps.itertuples(index=False):
        node = f"cap:{cid}"
        uf.find(node)
        if app:
            uf.union(node, f"app:{app}")
        if link_families and fam:
            uf.union(node, f"fam:{fam}")
    cap_group = {cid: "grp:" + uf.find(f"cap:{cid}") for cid in caps["capture_id"]}
    return df["capture_id"].map(cap_group)


# ----------------------------------------------------------------------------------- assignment
def _assign_class(groups: pd.DataFrame, fracs: Dict[str, float], rng: np.random.Generator,
                  forced: Dict[str, str]) -> Dict[str, str]:
    """groups: columns group_id, n (examples). Returns {group_id: split} for ONE class."""
    total = float(groups["n"].sum())
    target = {s: fracs[s] * total for s in SPLITS}
    cur = {s: 0.0 for s in SPLITS}
    out: Dict[str, str] = {}
    for gid, n in groups[["group_id", "n"]].itertuples(index=False):
        if gid in forced:
            out[gid] = forced[gid]
            cur[forced[gid]] += n
    rest = groups[~groups["group_id"].isin(out)].sort_values("group_id")
    order = rng.permutation(len(rest))
    for i in order:
        gid, n = rest.iloc[i]["group_id"], float(rest.iloc[i]["n"])
        # largest relative deficit first; train wins ties
        deficit = {s: (target[s] - cur[s]) / max(target[s], 1.0) for s in SPLITS}
        best = max(SPLITS, key=lambda s: (deficit[s], s == "train"))
        out[gid] = best
        cur[best] += n
    return out


def make_split(man: pd.DataFrame, strategy: str, seed: int, val_frac: float, test_frac: float,
               test_families: Sequence[str] = (), val_families: Sequence[str] = (),
               test_subsets: Sequence[str] = (), val_subsets: Sequence[str] = (),
               exclude_local_infra: bool = False, min_window_packets: int = 0) -> pd.DataFrame:
    """Return DataFrame[example_id, split, group_id, exclude_reason] (same row order as `man`)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("val_frac and test_frac must be in (0,1) with val_frac + test_frac < 1")
    if man["example_id"].duplicated().any():
        raise ValueError("manifest has duplicate example_id values")
    listed_f = set(test_families) | set(val_families)
    if listed_f and strategy != "family_holdout":
        raise ValueError("--test-families/--val-families require --strategy family_holdout")
    if listed_f - set(man["malware_family"]):
        raise ValueError(f"unknown families (typo?): {sorted(listed_f - set(man['malware_family']))}")
    listed_s = set(test_subsets) | set(val_subsets)
    if listed_s - set(man["subset"]):
        raise ValueError(f"unknown subsets: {sorted(listed_s - set(man['subset']))}")
    df = add_flow_tags(man) if exclude_local_infra else man.copy()
    df["group_id"] = build_groups(df, link_families=(strategy == "family_holdout"))
    reason = pd.Series("", index=df.index, dtype=object)
    reason[df["binary_label"] < 0] = "unknown_label"
    if exclude_local_infra:
        reason[(reason == "") & (df["flow_tag"] == "local_infra")] = "local_infra"
    if min_window_packets > 0:
        reason[(reason == "") & (df["n_packets"] < min_window_packets)] = "short_window"
    df["exclude_reason"] = reason
    elig = df[df["exclude_reason"] == ""]
    if elig.empty:
        raise ValueError("no eligible examples after filtering")
    fracs = {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    rng = np.random.default_rng(seed)
    assignment: Dict[str, str] = {}
    fam_force = {**{f: "val" for f in val_families}, **{f: "test" for f in test_families}}
    overlap = set(val_families) & set(test_families)
    if overlap:
        raise ValueError(f"families listed for both val and test: {sorted(overlap)}")
    sub_force = {**{f: "val" for f in val_subsets}, **{f: "test" for f in test_subsets}}
    for cls in sorted(elig["binary_label"].unique()):
        e = elig[elig["binary_label"] == cls]
        g = e.groupby("group_id").agg(n=("example_id", "size")).reset_index()
        forced: Dict[str, str] = {}
        if strategy == "family_holdout":
            for gid, sub in e.groupby("group_id"):
                fams = set(sub["malware_family"]) - {""}
                hit = {fam_force[f] for f in fams if f in fam_force}
                if len(hit) > 1:
                    raise ValueError(f"group {gid} links families assigned to different splits: {sorted(fams)}")
                if hit:
                    forced[gid] = hit.pop()
        for gid, sub in e.groupby("group_id"):
            sset = set(sub["subset"]) - {""}
            hit = {sub_force[s] for s in sset if s in sub_force}
            if hit and gid not in forced:
                if len(hit) > 1:
                    raise ValueError(f"group {gid} spans subsets assigned to different splits")
                forced[gid] = hit.pop()
        assignment.update(_assign_class(g, fracs, rng, forced))
    df["split"] = df["group_id"].map(assignment).where(df["exclude_reason"] == "", "excluded")
    return df[["example_id", "split", "group_id", "exclude_reason"]]


# ----------------------------------------------------------------------------------- checks
def check_leakage(man: pd.DataFrame, sp: pd.DataFrame, strategy: str) -> List[str]:
    """Return a list of violations (empty = clean). Only train/val/test rows are considered."""
    errs: List[str] = []
    d = man.merge(sp[["example_id", "split", "group_id"]], on="example_id", how="left", validate="one_to_one")
    if d["split"].isna().any():
        errs.append(f"{int(d['split'].isna().sum())} manifest examples have no split row")
    if len(sp) != len(man):
        errs.append("split table and manifest differ in length")
    used = d[d["split"].isin(SPLITS)]
    keys = ["group_id", "capture_id", "flow_id"]
    for key in keys:
        n = used.groupby(key)["split"].nunique()
        if (n > 1).any():
            errs.append(f"{key}: {int((n > 1).sum())} values appear in more than one split, e.g. {list(n[n > 1].index[:3])}")
    for key in ("app_id",) + (("malware_family",) if strategy == "family_holdout" else ()):
        u = used[used[key] != ""]
        n = u.groupby(key)["split"].nunique()
        if (n > 1).any():
            errs.append(f"{key}: {int((n > 1).sum())} values appear in more than one split, e.g. {list(n[n > 1].index[:3])}")
    if (used["binary_label"] < 0).any():
        errs.append("examples with binary_label = -1 (unknown) leaked into train/val/test")
    return errs


def summarise(man: pd.DataFrame, sp: pd.DataFrame) -> Dict[str, Any]:
    d = man.merge(sp[["example_id", "split", "exclude_reason"]], on="example_id")
    out: Dict[str, Any] = {"per_split": {}, "excluded": d.loc[d["split"] == "excluded", "exclude_reason"].value_counts().to_dict()}
    for s in (*SPLITS, "excluded"):
        x = d[d["split"] == s]
        if x.empty:
            out["per_split"][s] = {"examples": 0}
            continue
        entry = {
            "examples": int(len(x)), "flows": int(x["flow_id"].nunique()), "captures": int(x["capture_id"].nunique()),
            "apps": int(x.loc[x["app_id"] != "", "app_id"].nunique()),
            "families": sorted(x.loc[x["malware_family"] != "", "malware_family"].unique().tolist()),
            "label_counts": {str(k): int(v) for k, v in x["label"].value_counts().items()},
            "positive_rate": round(float((x["binary_label"] == 1).mean()), 4) if s != "excluded" else None,
        }
        dates = x.loc[x["capture_date"] != "", "capture_date"]
        if len(dates):
            entry["capture_dates"] = [dates.min(), dates.max()]
        out["per_split"][s] = entry
    return out


def to_markdown(args: Dict[str, Any], summary: Dict[str, Any], errs: List[str], warns: List[str]) -> str:
    L = [f"# Split `{args['name']}`", "", "```", json.dumps(args, indent=2), "```", "",
         "## Leakage checks", ""]
    L += [f"- FAIL: {e}" for e in errs] or ["- all checks passed (no group / capture / flow / app"
                                           + (" / family" if args["strategy"] == "family_holdout" else "")
                                           + " shared across train/val/test)"]
    L += ["", "## Warnings", ""] + ([f"- {w}" for w in warns] or ["- none"])
    L += ["", "## Per split", "", "| split | examples | flows | captures | apps | positive rate | label counts | dates |",
          "|---|---:|---:|---:|---:|---:|---|---|"]
    for s, e in summary["per_split"].items():
        if e["examples"] == 0:
            L.append(f"| {s} | 0 |  |  |  |  |  |  |")
            continue
        L.append(f"| {s} | {e['examples']} | {e['flows']} | {e['captures']} | {e['apps']} | {e['positive_rate']} | "
                 f"{e['label_counts']} | {e.get('capture_dates', '')} |")
    L += ["", f"Excluded examples by reason: {summary['excluded']}", "", "## Families per split", ""]
    for s in SPLITS:
        L.append(f"- **{s}**: {summary['per_split'][s].get('families', [])}")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--name", required=True, help="split name -> data/splits/<name>.csv")
    ap.add_argument("--strategy", choices=STRATEGIES, default="group_random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--test-families", default="", help="comma list; family_holdout only")
    ap.add_argument("--val-families", default="", help="comma list; family_holdout only")
    ap.add_argument("--test-subsets", default="", help="comma list of manifest `subset` values forced to test (e.g. 2017)")
    ap.add_argument("--val-subsets", default="")
    ap.add_argument("--exclude-local-infra", action="store_true",
                    help="drop windows whose responder is a private/link-local/multicast address")
    ap.add_argument("--min-window-packets", type=int, default=0)
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--allow-degenerate", action="store_true", help="do not fail if a split lacks a class")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    csv_list = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    try:
        header = pd.read_csv(a.manifest, nrows=0).columns
        man = load_manifest(a.manifest, usecols=[c for c in USECOLS if c in header])
        need = {"example_id", "capture_id", "app_id", "malware_family", "label", "binary_label", "flow_id", "n_packets"}
        if a.exclude_local_infra:
            need |= {"responder_ip", "responder_port", "ip_proto"}
        if need - set(man.columns):
            log.error("manifest lacks columns %s", sorted(need - set(man.columns)))
            return 2
        for col in ("subset", "capture_date"):
            if col not in man.columns:
                man[col] = ""
        sp = make_split(man, a.strategy, a.seed, a.val_frac, a.test_frac, csv_list(a.test_families),
                        csv_list(a.val_families), csv_list(a.test_subsets), csv_list(a.val_subsets),
                        a.exclude_local_infra, a.min_window_packets)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    errs = check_leakage(man, sp, a.strategy)
    summary = summarise(man, sp)
    warns: List[str] = []
    for s in SPLITS:
        e = summary["per_split"][s]
        if e["examples"] == 0:
            warns.append(f"split '{s}' is empty")
            continue
        if len(e["label_counts"]) < 2:
            warns.append(f"split '{s}' contains only {list(e['label_counts'])}")
    tot = sum(summary["per_split"][s]["examples"] for s in SPLITS) or 1
    for s, f in (("val", a.val_frac), ("test", a.test_frac)):
        got = summary["per_split"][s]["examples"] / tot
        if abs(got - f) > 0.05:
            warns.append(f"{s} holds {got:.1%} of examples (requested {f:.0%}): few large groups limit balance")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{a.name}.csv"
    sp.to_csv(csv_path, index=False)
    args = {**vars(a), "manifest_rows": int(len(man)),
            "split_csv_sha1": hashlib.sha1(csv_path.read_bytes()).hexdigest()}
    (out / f"{a.name}.json").write_text(json.dumps({"args": args, "summary": summary, "leakage_errors": errs,
                                                    "warnings": warns}, indent=2, default=str))
    (out / f"{a.name}.md").write_text(to_markdown(args, summary, errs, warns))
    print(f"split '{a.name}' written to {csv_path}")
    for s, e in summary["per_split"].items():
        if e["examples"]:
            print(f"  {s:9s} examples={e['examples']:8d} captures={e['captures']:5d} labels={e['label_counts']}")
    print(f"  excluded: {summary['excluded']}")
    for w in warns:
        print("  WARNING:", w)
    if errs:
        for e in errs:
            print("  LEAKAGE:", e)
        return 1
    print("  leakage checks: passed")
    degenerate = any(len(summary["per_split"][s].get("label_counts", {})) < 2 for s in SPLITS)
    return 1 if (degenerate and not a.allow_degenerate) else 0


if __name__ == "__main__":
    sys.exit(main())
