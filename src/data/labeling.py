"""Configurable, auditable label assignment.

Labels are assigned PER CAPTURE FILE from its relative path (and an optional override CSV).
That makes them weak, capture-level labels: "this capture was recorded while sample X ran".
They never prove an individual flow was malicious, which is why the mapped vocabulary has
`malicious_associated` and why nothing here emits `confirmed_attack` unless a rule or override
explicitly says so.

Every manifest row keeps: original_label (verbatim), mapped_label, label_source, reliability,
and the name of the rule that fired.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

log = logging.getLogger(__name__)

CATEGORIES = ("benign", "malicious_associated", "confirmed_attack", "unknown")


@dataclass
class LabelResult:
    rule_name: str
    original_label: str
    mapped_label: str
    label_source: str
    reliability: str
    app_id: str = ""
    malware_family: str = ""
    category: str = ""
    capture_date: str = ""
    subset: str = ""
    binary_label: int = -1


_DATE_RE = re.compile(r"^(\d{2})_(\d{2})_(\d{4})$")


def normalise_date(raw: str) -> str:
    """'06_14_2017' (MM_DD_YYYY, as used in AndMal2017 file names) -> '2017-06-14'; else unchanged."""
    m = _DATE_RE.match(raw or "")
    return f"{m.group(3)}-{m.group(1)}-{m.group(2)}" if m else (raw or "")


class LabelMapper:
    def __init__(self, config: Dict[str, Any], raw_root: Optional[Path] = None, warn: bool = True):
        self.cats = tuple(config.get("categories", CATEGORIES))
        if set(self.cats) != set(CATEGORIES):
            raise ValueError(f"labels.categories must be exactly {CATEGORIES}")
        self.binary_target: Dict[str, int] = dict(config.get("binary_target", {}))
        missing = [c for c in CATEGORIES if c not in self.binary_target]
        if missing:
            raise ValueError(f"labels.binary_target missing categories: {missing}")
        self.default = dict(config.get("default_rule") or {})
        self.default.setdefault("mapped_label", "unknown")
        self.rules: List[Dict[str, Any]] = []
        for r in config.get("rules") or []:
            if not r.get("enabled", False):
                continue
            for k in ("name", "path_regex", "mapped_label"):
                if k not in r:
                    raise ValueError(f"Label rule missing '{k}': {r}")
            if r["mapped_label"] not in CATEGORIES:
                raise ValueError(f"Rule {r['name']}: mapped_label must be one of {CATEGORIES}")
            self.rules.append({**r, "_re": re.compile(r["path_regex"])})
        self.overrides: Dict[str, Dict[str, Any]] = {}
        ov = config.get("override_csv")
        if ov:
            ov_path = Path(ov)
            if not ov_path.is_file():
                raise FileNotFoundError(f"labels.override_csv not found: {ov_path}")
            df = pd.read_csv(ov_path, dtype=str).fillna("")
            if "capture_path" not in df.columns:
                raise ValueError("override_csv must contain a 'capture_path' column")
            for rec in df.to_dict("records"):
                self.overrides[rec["capture_path"].replace("\\", "/")] = rec
        self.n_enabled_rules = len(self.rules)
        if warn and self.n_enabled_rules == 0 and not self.overrides:
            log.warning("No label rules are enabled and no override CSV is set: EVERY example will be "
                        "labeled 'unknown'. Edit configs/labels.yaml after reading the inspection report.")

    @classmethod
    def from_file(cls, path: str | Path, raw_root: Optional[Path] = None) -> "LabelMapper":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Label config not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh) or {}, raw_root)

    def label_capture(self, rel_path: str) -> LabelResult:
        rel_path = rel_path.replace("\\", "/")
        res: Optional[LabelResult] = None
        for rule in self.rules:
            m = rule["_re"].search(rel_path)
            if not m:
                continue
            g = m.groupdict()
            orig = rule.get("original_label", "")
            if orig == "from_directory":
                orig = g.get("family") or g.get("category") or ""
            res = LabelResult(
                rule_name=rule["name"], original_label=str(orig), mapped_label=rule["mapped_label"],
                label_source=rule.get("label_source", "path_rule"),
                reliability=rule.get("reliability", "unspecified"),
                app_id=g.get("app") or "", malware_family=g.get("family") or "",
                category=g.get("category") or "", capture_date=normalise_date(g.get("date") or ""),
                subset=g.get("subset") or "")
            break
        if res is None:
            res = LabelResult(rule_name="default", original_label="",
                              mapped_label=self.default["mapped_label"],
                              label_source=self.default.get("label_source", "no_rule_matched"),
                              reliability=self.default.get("reliability", "none"))
        ov = self.overrides.get(rel_path)
        if ov:
            if ov.get("mapped_label"):
                if ov["mapped_label"] not in CATEGORIES:
                    raise ValueError(f"override for {rel_path}: bad mapped_label {ov['mapped_label']!r}")
                res.mapped_label = ov["mapped_label"]
            for src, dst in (("original_label", "original_label"), ("app_id", "app_id"),
                             ("malware_family", "malware_family"), ("label_source", "label_source"),
                             ("reliability", "reliability"), ("capture_date", "capture_date"),
                             ("subset", "subset")):
                if ov.get(src):
                    setattr(res, dst, ov[src])
            res.rule_name = "override_csv"
        res.binary_label = int(self.binary_target[res.mapped_label])
        return res
