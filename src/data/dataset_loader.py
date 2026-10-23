"""Load processed shards + manifest back into memory (Phase 2 scope: loading and validation only).

Splitting, normalisation fitting and class balancing are Phase 4 and deliberately NOT here, so
that no statistic is ever computed on data that later becomes a test set.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.data.feature_extraction import AGG_FEATURE_NAMES, aggregate_features
from src.utils.config import META_CHANNELS

log = logging.getLogger(__name__)

ARRAY_KEYS = ("meta", "mask", "payload_valid_len", "rel_time", "payload")


def load_manifest(path: str | Path, usecols=None) -> pd.DataFrame:
    """Load a manifest CSV. `usecols` limits columns (the full manifest can be large)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    dtypes = {"warnings": str, "app_id": str, "malware_family": str, "capture_date": str, "subset": str,
              "malware_category": str, "original_label": str}
    if usecols is not None:
        dtypes = {k: v for k, v in dtypes.items() if k in usecols}
    return pd.read_csv(path, keep_default_na=False, dtype=dtypes, usecols=usecols)


def load_arrays(manifest: pd.DataFrame, processed_root: str | Path,
                keys: Iterable[str] = ARRAY_KEYS) -> Dict[str, np.ndarray]:
    """Return arrays aligned row-for-row with `manifest` (any row order/subset is honoured)."""
    root = Path(processed_root)
    keys = list(keys)
    out: Dict[str, list] = {k: [] for k in keys}
    spec_seen: Optional[str] = None
    for shard, grp in manifest.groupby("shard_file", sort=False):
        p = root / shard
        if not p.is_file():
            raise FileNotFoundError(f"Shard referenced by manifest is missing: {p}")
        with np.load(p, allow_pickle=False) as z:
            spec = str(z["feature_spec_json"])
            if spec_seen is None:
                spec_seen = spec
            elif spec != spec_seen:
                raise ValueError(f"Shard {p.name} has a different feature spec than earlier shards")
            rows = grp["shard_row"].to_numpy()
            ids = z["example_ids"][rows]
            if not np.array_equal(ids, grp["example_id"].to_numpy().astype(str)):
                raise ValueError(f"Manifest/shard example_id mismatch in {p.name}")
            for k in keys:
                if k in z.files:
                    out[k].append((grp.index.to_numpy(), z[k][rows]))
    result: Dict[str, np.ndarray] = {}
    pos = {idx: i for i, idx in enumerate(manifest.index)}
    for k, parts in out.items():
        if not parts:
            continue
        sample = parts[0][1]
        arr = np.zeros((len(manifest),) + sample.shape[1:], dtype=sample.dtype)
        for idxs, data in parts:
            arr[[pos[i] for i in idxs]] = data
        result[k] = arr
    result["spec"] = json.loads(spec_seen) if spec_seen else {}
    return result


def mlp_features(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """Model A input (raw aggregate features; normalisation is fitted on train only in Phase 4)."""
    return aggregate_features(arrays["meta"], arrays["mask"])


__all__ = ["load_manifest", "load_arrays", "mlp_features", "AGG_FEATURE_NAMES", "META_CHANNELS"]


def load_consolidated(out_dir: str | Path, split: str, mmap: bool = True) -> Dict[str, object]:
    """Open a split written by src.data.consolidate. Arrays are raw (un-normalised) numpy
    (memory-mapped by default); `index` is the per-row metadata DataFrame; `y` are int8 labels."""
    d = Path(out_dir)
    if not (d / f"{split}_index.csv").is_file():
        raise FileNotFoundError(f"split '{split}' not found in {d}")
    mode = "r" if mmap else None
    res: Dict[str, object] = {"index": pd.read_csv(d / f"{split}_index.csv", keep_default_na=False),
                              "y": np.load(d / f"{split}_y.npy")}
    for k in ARRAY_KEYS:
        p = d / f"{split}_{k}.npy"
        if p.exists():
            res[k] = np.load(p, mmap_mode=mode)
    return res
