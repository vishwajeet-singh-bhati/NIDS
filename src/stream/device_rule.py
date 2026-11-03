"""Device-level alarm: raise ONE alarm when >= N flow alerts happen within T seconds on the same device.

Why: labels are capture-level, so a single flagged flow is weak evidence (malware captures contain lots of
ordinary background traffic, benign captures contain the occasional odd flow). A burst of flagged flows in a
short time is far stronger evidence and is the natural "flag this device" decision for a monitoring app.
The same class is used online (engine) and offline (replay evaluation) so they cannot diverge.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional


@dataclass
class BurstParams:
    window_s: float = 60.0
    min_alerts: int = 10
    cooldown_s: Optional[float] = None      # None -> window_s: after an alarm, stay quiet that long


class BurstDetector:
    def __init__(self, p: BurstParams):
        self.p = p
        self.times: Deque[float] = deque()
        self.quiet_until = float("-inf")

    def add(self, t: float) -> bool:
        """Register a flow alert at time t (non-decreasing). True if this one triggers a device alarm."""
        self.times.append(t)
        while self.times and t - self.times[0] > self.p.window_s:
            self.times.popleft()
        if len(self.times) >= self.p.min_alerts and t >= self.quiet_until:
            self.quiet_until = t + (self.p.window_s if self.p.cooldown_s is None else self.p.cooldown_s)
            return True
        return False


def alarm_times(alert_times: Iterable[float], p: BurstParams) -> List[float]:
    d = BurstDetector(p)
    return [t for t in sorted(alert_times) if d.add(t)]
