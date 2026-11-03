"""Phase 4: turn per-capture shards + a split file into per-split memory-mapped arrays, then fit the
train-only normaliser.

    python -m src.data.consolidate --manifest data/manifests/cic_andmal2017_manifest.csv \
        --processed-root data/processed/cic_andmal2017 --splits data/splits/group_random_s0.csv \
        --out data/datasets/group_random_s0 [--no-payload]

Output directory:
    {train,val,test}_meta.npy / _mask.npy / _payload_valid_len.npy / _rel_time.npy / [_payload.npy]
    {train,val,test}_y.npy        int8 binary labels (0 benign, 1 malicious_associated)
    {train,val,test}_index.csv    example_id, binary_label, label, group_id, capture_id, app_id,
                                  malware_family, malware_category, capture_date, subset, flow_id, n_packets,
                                  responder_scope, service, flow_tag   (row i of the index == row i of the arrays)
    normalization.json            fitted on TRAIN ONLY
    dataset_card.json             shapes, counts, spec, hashes
Arrays are raw (un-normalised); apply `Normalizer.transform_*` at load time. Use
`dataset_loader.load_consolidated(dir, split)` to open them (memory-mapped by default).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.dataset_loader import load_manifest
from src.data.flow_tags import add_flow_tags
from src.data.normalization import Normalizer
from src.data.splits import SPLITS
from src.utils.logging_utils import setup_logging

log = logging.getLogger("consolidate")

INDEX_COLS = ["example_id", "binary_label", "label", "group_id", "capture_id", "app_id", "malware_family",
              "malware_category", "capture_date", "subset", "flow_id", "n_packets", "responder_scope", "service",
              "flow_tag"]
MAN_COLS = ["example_id", "capture_id", "app_id", "malware_family", "malware_category", "capture_date", "subset",
            "label", "binary_label", "flow_id", "n_packets", "ip_proto", "responder_ip", "responder_port",
            "shard_file", "shard_row"]


def consolidate(man: pd.DataFrame, sp: pd.DataFrame, processed_root: Path, out: Path, include_payload: bool,
                chunk: int = 50_000) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    d = man.merge(sp[["example_id", "split", "group_id"]], on="example_id", how="inner", validate="one_to_one")
    if len(d) != len(man):
        raise ValueError(f"split file covers {len(d)} of {len(man)} manifest examples")
    d = add_flow_tags(d)
    d = d[d["split"].isin(SPLITS)].reset_index(drop=True)
    if d.empty:
        raise ValueError("no train/val/test examples")
    # probe one shard for shapes/dtypes/spec
    first = processed_root / d["shard_file"].iloc[0]
    with np.load(first, allow_pickle=False) as z:
        spec = json.loads(str(z["feature_spec_json"]))
        keys = ["meta", "mask", "payload_valid_len", "rel_time"] + (["payload"] if "payload" in z.files and include_payload else [])
        shapes = {k: z[k].shape[1:] for k in keys}
        dtypes = {k: z[k].dtype for k in keys}
    if include_payload and "payload" not in keys:
        log.warning("shards contain no payload; continuing without it")
    maps: Dict[str, Dict[str, np.ndarray]] = {}
    pos: Dict[str, Dict[str, int]] = {}
    idx_frames: Dict[str, pd.DataFrame] = {}
    for s in SPLITS:
        ds = d[d["split"] == s].reset_index(drop=True)
        idx_frames[s] = ds
        pos[s] = {eid: i for i, eid in enumerate(ds["example_id"])}
        if ds.empty:
            continue
        maps[s] = {k: np.lib.format.open_memmap(out / f"{s}_{k}.npy", mode="w+", dtype=dtypes[k],
                                                shape=(len(ds),) + shapes[k]) for k in keys}
        np.save(out / f"{s}_y.npy", ds["binary_label"].to_numpy(dtype=np.int8))
        ds[[c for c in INDEX_COLS if c in ds.columns]].to_csv(out / f"{s}_index.csv", index=False)
    done = {s: 0 for s in SPLITS}
    for shard, g in d.groupby("shard_file", sort=True):
        with np.load(processed_root / shard, allow_pickle=False) as z:
            if json.loads(str(z["feature_spec_json"])) != spec:
                raise ValueError(f"shard {shard} has a different feature spec")
            ids = z["example_ids"]
            arrays = {k: z[k] for k in keys}
        for s, gs in g.groupby("split"):
            rows = gs["shard_row"].to_numpy()
            if not np.array_equal(ids[rows], gs["example_id"].to_numpy().astype(str)):
                raise ValueError(f"manifest/shard example mismatch in {shard}")
            dst = np.array([pos[s][e] for e in gs["example_id"]])
            for k in keys:
                maps[s][k][dst] = arrays[k][rows]
            done[s] += len(gs)
    for s in maps:
        for m in maps[s].values():
            m.flush()
    if any(done[s] != len(idx_frames[s]) for s in SPLITS):
        raise RuntimeError(f"rows written {done} != expected {({s: len(idx_frames[s]) for s in SPLITS})}")
    # ---- fit normaliser on TRAIN ONLY
    tr = np.load(out / "train_meta.npy", mmap_mode="r"), np.load(out / "train_mask.npy", mmap_mode="r")

    def chunks():
        for i in range(0, tr[0].shape[0], chunk):
            yield np.asarray(tr[0][i:i + chunk]), np.asarray(tr[1][i:i + chunk])

    norm = Normalizer.fit(chunks(), idx_frames["train"]["example_id"].tolist(), spec, fit_split="train")
    norm.save(out / "normalization.json")
    card = {"feature_spec": spec, "arrays": {k: {"shape_tail": list(shapes[k]), "dtype": str(dtypes[k])} for k in keys},
            "splits": {s: {"examples": int(len(idx_frames[s])),
                           "label_counts": {str(k): int(v) for k, v in idx_frames[s]["label"].value_counts().items()}}
                       for s in SPLITS},
            "normalization_fit": norm.spec["fit"], "include_payload": "payload" in keys}
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2))
    return card


def verify(out: Path, processed_root: Path, man: pd.DataFrame, n_per_split: int = 5, seed: int = 0) -> List[str]:
    """Spot-check consolidated rows against the original shards. Returns a list of problems."""
    problems: List[str] = []
    rng = np.random.default_rng(seed)
    byid = man.set_index("example_id")
    for s in SPLITS:
        ip = out / f"{s}_index.csv"
        if not ip.exists():
            continue
        idx = pd.read_csv(ip, keep_default_na=False)
        if idx.empty:
            continue
        arr = {k: np.load(out / f"{s}_{k}.npy", mmap_mode="r") for k in ("meta", "mask", "payload_valid_len", "rel_time")}
        pay = np.load(out / f"{s}_payload.npy", mmap_mode="r") if (out / f"{s}_payload.npy").exists() else None
        y = np.load(out / f"{s}_y.npy")
        if len(y) != len(idx) or arr["meta"].shape[0] != len(idx):
            problems.append(f"{s}: length mismatch")
            continue
        for i in rng.choice(len(idx), size=min(n_per_split, len(idx)), replace=False):
            eid = idx["example_id"].iloc[i]
            r = byid.loc[eid]
            with np.load(processed_root / r["shard_file"], allow_pickle=False) as z:
                row = int(r["shard_row"])
                if str(z["example_ids"][row]) != eid:
                    problems.append(f"{s}:{eid}: shard row id mismatch")
                    continue
                for k in arr:
                    if not np.array_equal(arr[k][i], z[k][row]):
                        problems.append(f"{s}:{eid}: array '{k}' differs from shard")
                if pay is not None and not np.array_equal(pay[i], z["payload"][row]):
                    problems.append(f"{s}:{eid}: payload differs from shard")
            if int(y[i]) != int(r["binary_label"]):
                problems.append(f"{s}:{eid}: label differs from manifest")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--splits", required=True, help="CSV from src.data.splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-payload", action="store_true", help="skip the (large) payload arrays")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    try:
        header = pd.read_csv(a.manifest, nrows=0).columns
        man = load_manifest(a.manifest, usecols=[c for c in MAN_COLS if c in header])
        for col in ("subset", "capture_date", "malware_category"):
            if col not in man.columns:
                man[col] = ""
        sp = pd.read_csv(a.splits, keep_default_na=False)
        card = consolidate(man, sp, Path(a.processed_root), Path(a.out), include_payload=not a.no_payload)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2
    problems = verify(Path(a.out), Path(a.processed_root), man)
    for s, e in card["splits"].items():
        print(f"  {s:5s} examples={e['examples']:8d} labels={e['label_counts']}")
    print(f"  normalizer fitted on {card['normalization_fit']['n_examples']} TRAIN examples "
          f"({card['normalization_fit']['n_real_packets']} real packets) -> {a.out}/normalization.json")
    if problems:
        for p in problems:
            print("  VERIFY FAIL:", p)
        return 1
    print("  spot-check against original shards: passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
