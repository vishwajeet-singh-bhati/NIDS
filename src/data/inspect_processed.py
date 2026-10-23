"""Inspect and VALIDATE processed tensors.

    python -m src.data.inspect_processed --manifest data/manifests/cic_andmal2017_manifest.csv \
        --processed-root data/processed/cic_andmal2017 --show 3

Prints dimensions, label distributions, mask statistics and a few decoded examples, then runs
invariant checks. Exit code 1 if any invariant fails.

Invariants checked
  I1  mask is binary and "left-packed" (all real packets precede all padding)
  I2  meta rows of padding packets are exactly zero
  I3  direction is +/-1 on real packets
  I4  first real packet has log_iat == 0 and rel_time == 0; rel_time is non-decreasing
  I5  exactly one protocol one-hot is set on each real packet
  I6  payload_valid_len == 0 on padding; <= B on real packets and <= payload_len
  I7  payload bytes beyond payload_valid_len are zero (padding never looks like data)
  I8  manifest n_packets equals mask.sum() for every row
  I9  no NaN/inf in meta
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.dataset_loader import load_arrays, load_manifest
from src.utils.config import META_CHANNELS
from src.utils.logging_utils import setup_logging

log = logging.getLogger("inspect_processed")
CH = {n: i for i, n in enumerate(META_CHANNELS)}


def check_invariants(arr: dict, manifest: pd.DataFrame) -> List[str]:
    errs: List[str] = []
    meta, mask = arr["meta"], arr["mask"]
    N, W, _ = meta.shape
    if not np.isin(mask, (0, 1)).all():
        errs.append("I1: mask has values other than 0/1")
    n_real = mask.sum(1).astype(int)
    left_packed = (mask == (np.arange(W)[None, :] < n_real[:, None])).all()
    if not left_packed:
        errs.append("I1: mask is not left-packed")
    pad = mask == 0
    if np.abs(meta[pad]).max(initial=0) != 0:
        errs.append("I2: non-zero meta in padding rows")
    d = meta[:, :, CH["direction"]]
    if not np.isin(d[mask == 1], (-1.0, 1.0)).all():
        errs.append("I3: direction not +/-1 on real packets")
    if (meta[:, 0, CH["log_iat"]] != 0).any() or (arr["rel_time"][:, 0] != 0).any():
        errs.append("I4: first packet log_iat/rel_time not 0")
    rt = arr["rel_time"]
    if ((np.diff(rt, axis=1) < -1e-9) & (mask[:, 1:] == 1)).any():
        errs.append("I4: rel_time decreases inside a window")
    onehot = sum(meta[:, :, CH[c]] for c in ("proto_tcp", "proto_udp", "proto_dns", "proto_other"))
    if not (onehot[mask == 1] == 1).all():
        errs.append("I5: protocol one-hot not exactly one on some real packets")
    pvl = arr["payload_valid_len"]
    if (pvl[pad] != 0).any():
        errs.append("I6: payload_valid_len non-zero on padding")
    pl = meta[:, :, CH["payload_len"]]
    if (pvl[mask == 1] > pl[mask == 1]).any():
        errs.append("I6: payload_valid_len exceeds declared payload_len")
    if "payload" in arr:
        B = arr["payload"].shape[2]
        if (pvl > B).any():
            errs.append("I6: payload_valid_len > B")
        beyond = np.arange(B)[None, None, :] >= pvl[:, :, None]
        if (arr["payload"][beyond] != 0).any():
            errs.append("I7: non-zero payload bytes beyond payload_valid_len")
    if not np.array_equal(n_real, manifest["n_packets"].to_numpy()):
        errs.append("I8: manifest n_packets != mask.sum()")
    if not np.isfinite(meta).all():
        errs.append("I9: NaN/inf in meta")
    return errs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--show", type=int, default=3, help="number of example windows to print")
    ap.add_argument("--show-packets", type=int, default=8, help="packets printed per example")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-level", default="WARNING")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)
    try:
        man = load_manifest(a.manifest)
        arr = load_arrays(man, a.processed_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    spec = arr.pop("spec")
    meta, mask = arr["meta"], arr["mask"]
    print("=== Shapes ===")
    for k, v in arr.items():
        print(f"  {k:18s} {str(v.shape):20s} {v.dtype}")
    print(f"  feature_spec_version={spec.get('feature_spec_version')} window={spec.get('window_size')} "
          f"stride={spec.get('stride')} payload_bytes={spec.get('payload_bytes_kept')} "
          f"include_payload={spec.get('include_payload')}")
    print(f"  meta channels: {META_CHANNELS}")
    print("\n=== Manifest ===")
    print(f"  rows={len(man)} captures={man['capture_id'].nunique()} flows={man['flow_id'].nunique()}")
    print("  label counts:\n" + man["label"].value_counts().to_string().replace("\n", "\n    "))
    print("  binary_label counts (-1 = excluded):", man["binary_label"].value_counts().to_dict())
    print("  label_reliability:", man["label_reliability"].value_counts().to_dict())
    print("  protocol of flow (ip_proto):", man["ip_proto"].value_counts().to_dict())
    print("\n=== Packets per window ===")
    n = mask.sum(1)
    print(f"  min={n.min()} mean={n.mean():.2f} max={n.max()}  full windows={(n == mask.shape[1]).mean():.1%} "
          f"padded windows={(n < mask.shape[1]).mean():.1%}")
    print("  histogram of real packets:", np.bincount(n.astype(int), minlength=mask.shape[1] + 1).tolist())
    wcount = man["warnings"].replace("", np.nan).dropna().str.split(";").explode().value_counts()
    print("\n=== Processing warnings (examples affected) ===")
    print("  " + (wcount.to_string().replace("\n", "\n  ") if len(wcount) else "none"))
    rng = np.random.default_rng(a.seed)
    idxs = rng.choice(len(man), size=min(a.show, len(man)), replace=False)
    for i in idxs:
        r = man.iloc[i]
        print(f"\n--- example {r['example_id']} | label={r['label']} (orig='{r['original_label']}', "
              f"rule={r['label_rule']}) | packets={r['n_packets']} | warnings={r['warnings'] or '-'}")
        print(f"    initiator={r['initiator_ip']}:{r['initiator_port']} responder={r['responder_ip']}:"
              f"{r['responder_port']} (audit only; never a model input)")
        print("    #  m dir  plen  logIAT  rel_s  proto flags  payload[:12] (valid_len)")
        for j in range(min(a.show_packets, mask.shape[1])):
            m = meta[i, j]
            proto = next((c[6:] for c in ("proto_tcp", "proto_udp", "proto_dns", "proto_other")
                          if m[CH[c]] == 1), "pad")
            flags = "".join(ch for ch, c in zip("SFRPA", ("flag_syn", "flag_fin", "flag_rst", "flag_psh",
                                                           "flag_ack")) if m[CH[c]] == 1) or "-"
            pay = arr["payload"][i, j, :12].tolist() if "payload" in arr else "n/a"
            print(f"   {j:2d} {int(mask[i, j])} {m[CH['direction']]:+.0f} {m[CH['payload_len']]:5.0f} "
                  f"{m[CH['log_iat']]:7.3f} {arr['rel_time'][i, j]:6.3f} {proto:5s} {flags:5s} {pay} "
                  f"({int(arr['payload_valid_len'][i, j])})")
    errs = check_invariants(arr, man)
    print("\n=== Invariant checks ===")
    if errs:
        for e in errs:
            print("  FAIL", e)
        return 1
    print("  all invariants I1-I9 passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
