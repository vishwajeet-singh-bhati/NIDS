"""Train-only normalisation, saved as JSON so Python training and the Android runtime share it.

Exact formulas (the Android implementation MUST reproduce these; Phase 7/8 adds a parity test):

  meta tensor  x[w, c]  (w = packet position, c = channel), mask m[w] in {0,1}
      transform "none":          y = x
      transform "zscore":        y = (x - mean_c) / std_c
      transform "log1p_zscore":  y = (log1p(x) - mean_c) / std_c
      then every padding row is zeroed:  y[w, :] = 0  when m[w] = 0
  channels: direction -> none, payload_len -> log1p_zscore, log_iat -> zscore (it is already
  log1p(ms)), all one-hot / flag channels -> none.
  mean_c / std_c are computed over REAL packets (mask = 1) of the TRAIN split only, in float64.
  std_c is floored at `std_floor` (1e-6).

  aggregate features (Model A) a[k]:  y = (t_k(a) - mean_k) / std_k with t_k = log1p for count/size
  features and identity for fractions/log-time statistics (listed in the JSON).

  payload bytes: p / 255.0 (no fitted statistics). Padding is identified with `payload_valid_len`
  (or the packet mask), not by the byte value.

Nothing here ever sees validation or test data: `fit` takes iterators over train arrays only, and
the saved JSON records the SHA-1 of the sorted train example ids used for fitting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from src.data.feature_extraction import AGG_FEATURE_NAMES, aggregate_features
from src.utils.config import META_CHANNELS

NORMALIZATION_VERSION = "1"
STD_FLOOR = 1e-6

META_TRANSFORMS: Dict[str, str] = {c: "none" for c in META_CHANNELS}
META_TRANSFORMS.update({"payload_len": "log1p_zscore", "log_iat": "zscore"})

AGG_LOG1P = {"n_packets", "n_fwd", "n_rev", "sum_payload_fwd", "sum_payload_rev", "mean_payload_fwd",
             "mean_payload_rev", "max_payload", "std_payload", "n_syn", "n_fin", "n_rst", "n_psh"}
AGG_TRANSFORMS: List[str] = ["log1p" if n in AGG_LOG1P else "none" for n in AGG_FEATURE_NAMES]


class _Acc:
    """float64 streaming mean/variance."""

    def __init__(self, k: int):
        self.n = 0.0
        self.s = np.zeros(k)
        self.ss = np.zeros(k)

    def add(self, x: np.ndarray) -> None:  # x [N, k]
        x = x.astype(np.float64, copy=False)
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += (x * x).sum(0)

    def finish(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.s / max(self.n, 1.0)
        var = np.maximum(self.ss / max(self.n, 1.0) - mean ** 2, 0.0)
        return mean, np.maximum(np.sqrt(var), STD_FLOOR)


def _meta_pre(meta: np.ndarray) -> np.ndarray:
    """Apply the log1p pre-transforms (float64) to a copy of meta."""
    x = meta.astype(np.float64, copy=True)
    for i, c in enumerate(META_CHANNELS):
        if META_TRANSFORMS[c] == "log1p_zscore":
            x[..., i] = np.log1p(x[..., i])
    return x


class Normalizer:
    def __init__(self, spec: Dict[str, Any]):
        if spec.get("normalization_version") != NORMALIZATION_VERSION:
            raise ValueError(f"unsupported normalization version {spec.get('normalization_version')}")
        if spec["meta_channels"] != META_CHANNELS:
            raise ValueError("normalizer was fitted for a different meta channel layout")
        self.spec = spec
        ch = spec["meta"]["channels"]
        self._mean = np.array([ch[c]["mean"] for c in META_CHANNELS], dtype=np.float64)
        self._std = np.array([ch[c]["std"] for c in META_CHANNELS], dtype=np.float64)
        self._tr = [ch[c]["transform"] for c in META_CHANNELS]
        ag = spec["aggregate"]
        self._amean = np.array(ag["mean"], dtype=np.float64)
        self._astd = np.array(ag["std"], dtype=np.float64)
        self._atr = ag["transforms"]

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, chunks: Iterable[Tuple[np.ndarray, np.ndarray]], example_ids: Sequence[str],
            feature_spec: Dict[str, Any], fit_split: str = "train") -> "Normalizer":
        """chunks yields (meta [N,W,C], mask [N,W]) for the TRAIN split only."""
        meta_acc = _Acc(len(META_CHANNELS))
        agg_acc = _Acc(len(AGG_FEATURE_NAMES))
        n_ex = 0
        for meta, mask in chunks:
            real = np.asarray(mask).astype(bool)
            pre = _meta_pre(np.asarray(meta))
            meta_acc.add(pre[real])                                   # [n_real_packets, C]
            agg_acc.add(cls._agg_pre(aggregate_features(np.asarray(meta), np.asarray(mask))))
            n_ex += meta.shape[0]
        if n_ex == 0:
            raise ValueError("cannot fit a normalizer on zero examples")
        m_mean, m_std = meta_acc.finish()
        a_mean, a_std = agg_acc.finish()
        spec = {
            "normalization_version": NORMALIZATION_VERSION,
            "feature_spec_version": feature_spec.get("feature_spec_version"),
            "window_size": feature_spec.get("window_size"),
            "meta_channels": list(META_CHANNELS),
            "meta": {"channels": {c: {"transform": META_TRANSFORMS[c],
                                      "mean": float(m_mean[i]) if META_TRANSFORMS[c] != "none" else 0.0,
                                      "std": float(m_std[i]) if META_TRANSFORMS[c] != "none" else 1.0}
                                  for i, c in enumerate(META_CHANNELS)},
                     "zero_padding_rows": True},
            "aggregate": {"names": list(AGG_FEATURE_NAMES), "transforms": AGG_TRANSFORMS,
                          "mean": [float(v) for v in a_mean], "std": [float(v) for v in a_std]},
            "payload": {"scale": 255.0},
            "std_floor": STD_FLOOR,
            "fit": {"split": fit_split, "n_examples": int(n_ex), "n_real_packets": int(meta_acc.n),
                    "example_ids_sha1": hashlib.sha1("\n".join(sorted(map(str, example_ids))).encode()).hexdigest()},
        }
        return cls(spec)

    @staticmethod
    def _agg_pre(agg: np.ndarray) -> np.ndarray:
        x = agg.astype(np.float64, copy=True)
        for k, t in enumerate(AGG_TRANSFORMS):
            if t == "log1p":
                x[:, k] = np.log1p(x[:, k])
        return x

    # ---------------------------------------------------------------- applying
    def transform_meta(self, meta: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """meta [..., W, C], mask [..., W] -> float32 normalised meta (padding rows = 0)."""
        x = _meta_pre(meta)
        for i, t in enumerate(self._tr):
            if t != "none":
                x[..., i] = (x[..., i] - self._mean[i]) / self._std[i]
        x *= np.asarray(mask, dtype=np.float64)[..., None]
        return x.astype(np.float32)

    def transform_aggregate(self, agg: np.ndarray) -> np.ndarray:
        x = self._agg_pre(np.atleast_2d(agg))
        return ((x - self._amean) / self._astd).astype(np.float32)

    @staticmethod
    def transform_payload(payload: np.ndarray) -> np.ndarray:
        return (payload.astype(np.float32) / 255.0)

    # ---------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.spec, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        return cls(json.loads(Path(path).read_text()))
