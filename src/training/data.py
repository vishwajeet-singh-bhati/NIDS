"""Batch feeder over a consolidated split (src.data.consolidate output).

* Arrays stay memory-mapped; a batch is gathered, optionally truncated to a packet PREFIX,
  normalised with the TRAIN-fitted `Normalizer`, and turned into torch tensors.
* The normaliser always comes from the training dataset (checkpoint), never from the split being
  evaluated.
* Class balancing (if requested) happens only in `iter_train_batches`, by resampling indices inside the
  training loop - the stored splits are never duplicated.

Batch dict keys (depending on `kinds`):
    mask  [B,W] float32        always
    meta  [B,W,12] float32     normalised, padding rows zero
    agg   [B,22]  float32      normalised aggregate features (Model A)
    payload [B,W,Bn] int64     byte+1, 0 = padding / not-captured  (Model C)
    y     [B] float32
    idx   [B] int64            row numbers inside the split (for bookkeeping)
"""
from __future__ import annotations

from typing import Dict, Iterator, Optional, Sequence, Set

import numpy as np
import torch

from src.data.dataset_loader import load_consolidated
from src.data.feature_extraction import aggregate_features
from src.data.normalization import Normalizer


class SplitData:
    def __init__(self, dataset_dir: str, split: str, normalizer: Normalizer, mmap: bool = True,
                 max_examples: int = 0, seed: int = 0):
        raw = load_consolidated(dataset_dir, split, mmap=mmap)
        self.dir, self.split, self.norm = dataset_dir, split, normalizer
        self.index = raw["index"]
        self.y_all: np.ndarray = raw["y"]
        self.meta, self.mask_raw = raw["meta"], raw["mask"]
        self.pvl = raw.get("payload_valid_len")
        self.rel_time = raw.get("rel_time")
        self.payload = raw.get("payload")
        n = len(self.y_all)
        if max_examples and n > max_examples:
            self.sel = np.sort(np.random.default_rng(seed).choice(n, size=max_examples, replace=False))
        else:
            self.sel = np.arange(n)
        spec_w = normalizer.spec.get("window_size")
        if spec_w is not None and self.meta.shape[1] != spec_w:
            raise ValueError(f"{dataset_dir}/{split}: window {self.meta.shape[1]} != normalizer window {spec_w}")

    # ----------------------------------------------------------------------------- info
    def __len__(self) -> int:
        return int(len(self.sel))

    @property
    def y(self) -> np.ndarray:
        return self.y_all[self.sel]

    @property
    def sel_index(self):
        return self.index.iloc[self.sel].reset_index(drop=True)

    @property
    def window_size(self) -> int:
        return int(self.meta.shape[1])

    @property
    def payload_bytes(self) -> int:
        return int(self.payload.shape[2]) if self.payload is not None else 0

    # ----------------------------------------------------------------------------- batches
    def batch(self, rows: np.ndarray, kinds: Set[str], prefix_k: Optional[int] = None) -> Dict[str, torch.Tensor]:
        rows = np.asarray(rows)
        order = np.argsort(rows, kind="stable")           # sorted reads are friendlier to memmaps
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        srt = rows[order]
        meta = np.asarray(self.meta[srt])[inv]
        mask = np.asarray(self.mask_raw[srt])[inv].copy()
        if prefix_k is not None:
            mask[:, prefix_k:] = 0
            meta = meta.copy()
            meta[:, prefix_k:] = 0
        out: Dict[str, torch.Tensor] = {"mask": torch.from_numpy(mask.astype(np.float32)),
                                        "y": torch.from_numpy(self.y_all[rows].astype(np.float32)),
                                        "idx": torch.from_numpy(rows.astype(np.int64))}
        if "meta" in kinds:
            out["meta"] = torch.from_numpy(self.norm.transform_meta(meta, mask))
        if "agg" in kinds:
            out["agg"] = torch.from_numpy(self.norm.transform_aggregate(aggregate_features(meta, mask)))
        if "payload" in kinds:
            if self.payload is None or self.pvl is None:
                raise ValueError("this dataset was consolidated without payload arrays (--no-payload)")
            pay = np.asarray(self.payload[srt])[inv].astype(np.int64)
            pvl = np.asarray(self.pvl[srt])[inv].astype(np.int64)
            valid = np.arange(pay.shape[2])[None, None, :] < pvl[:, :, None]
            valid &= (mask[:, :, None] > 0)
            out["payload"] = torch.from_numpy(np.where(valid, pay + 1, 0))
        return out

    def iter_eval_batches(self, batch_size: int, kinds: Set[str], prefix_k: Optional[int] = None
                          ) -> Iterator[Dict[str, torch.Tensor]]:
        for i in range(0, len(self.sel), batch_size):
            yield self.batch(self.sel[i:i + batch_size], kinds, prefix_k)

    def iter_train_batches(self, batch_size: int, kinds: Set[str], rng: np.random.Generator,
                           sampler: str = "none", max_steps: int = 0) -> Iterator[Dict[str, torch.Tensor]]:
        if sampler == "balanced":
            y = self.y_all[self.sel]
            n_pos, n_neg = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
            p = np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg)
            p = p / p.sum()
            order = rng.choice(self.sel, size=len(self.sel), replace=True, p=p)
        elif sampler == "none":
            order = rng.permutation(self.sel)
        else:
            raise ValueError("sampler must be 'none' or 'balanced'")
        steps = 0
        for i in range(0, len(order), batch_size):
            rows = order[i:i + batch_size]
            if len(rows) < 2:        # BatchNorm needs > 1 sample
                break
            yield self.batch(rows, kinds)
            steps += 1
            if max_steps and steps >= max_steps:
                break
