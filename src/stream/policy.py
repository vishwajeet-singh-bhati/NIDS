"""Flow-level alert policy: a pure function of the sequence of window logits of ONE flow.

Kept separate (and free of I/O) so the live engine and the offline replay evaluation apply exactly the
same rule: thresholds can be swept on recorded score traces without re-running the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass
class PolicyParams:
    threshold: float = 0.0     # logit
    ema_alpha: float = 0.5
    min_hits: int = 2          # consecutive windows >= threshold
    hard_margin: float = 3.0   # a single window >= threshold + margin alerts at once


@dataclass
class PolicyState:
    ema: Optional[float] = None
    hits: int = 0
    scored: int = 0
    alerted: bool = False


def step(st: PolicyState, z: float, p: PolicyParams) -> Optional[str]:
    """Update `st` with a new window logit; return an alert reason the first time the flow trips, else None."""
    st.scored += 1
    st.ema = z if st.ema is None else p.ema_alpha * z + (1 - p.ema_alpha) * st.ema
    st.hits = st.hits + 1 if z >= p.threshold else 0
    if st.alerted:
        return None
    reason = None
    if z >= p.threshold + p.hard_margin:
        reason = "single_window_high"
    elif st.hits >= p.min_hits and st.ema >= p.threshold:
        reason = "sustained"
    if reason:
        st.alerted = True
    return reason


def first_alert(logits: Sequence[float], p: PolicyParams) -> Tuple[Optional[int], Optional[str]]:
    """Index (into `logits`) of the window at which the flow first alerts, and why. (None, None) if never."""
    st = PolicyState()
    for i, z in enumerate(logits):
        r = step(st, float(z), p)
        if r:
            return i, r
    return None, None
