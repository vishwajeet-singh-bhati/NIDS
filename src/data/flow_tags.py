"""Rule-based, label-independent descriptors of WHAT a flow is talking to.

Why this exists: in CIC-AndMal2017 a "malware" label is attached to a whole capture, but a large
share of the traffic in that capture is generic phone/network chatter (DNS to the gateway, mDNS,
router discovery...) that looks the same in benign captures. These tags let us (a) measure how much
of each class is such background and (b) optionally run a primary experiment without it.

Everything here is computed from manifest columns only (responder_ip, responder_port, ip_proto),
never from labels, so using it as a filter cannot leak label information.
"""
from __future__ import annotations

import ipaddress
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd

SCOPES = ("public", "private", "link_local", "multicast", "loopback", "unspecified", "unparsable")


@lru_cache(maxsize=1_000_000)
def ip_scope(ip: str) -> str:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return "unparsable"
    if a.is_loopback:
        return "loopback"
    if a.is_multicast:
        return "multicast"
    if a.is_link_local:
        return "link_local"
    if a.is_unspecified:
        return "unspecified"
    if a.is_private or not a.is_global:   # RFC1918/ULA, CGNAT 100.64/10, documentation ranges, ...
        return "private"
    return "public"


# (ip_proto, port) -> service hint. Deliberately small and conventional; "other" is the default.
_SERVICES: Dict[tuple, str] = {
    (17, 53): "dns", (6, 53): "dns", (17, 5353): "mdns", (17, 123): "ntp", (17, 67): "dhcp",
    (17, 68): "dhcp", (17, 1900): "ssdp", (17, 137): "netbios", (17, 138): "netbios",
    (6, 443): "https", (6, 80): "http", (17, 443): "quic", (6, 5228): "gcm_fcm", (6, 853): "dot",
}


def service_hint(ip_proto: int, port: int) -> str:
    if ip_proto in (1, 58):
        return "icmp"
    return _SERVICES.get((int(ip_proto), int(port)), "other")


def add_flow_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with extra columns: responder_scope, service, flow_tag.

    flow_tag = "local_infra" when the responder is NOT a public address (private/link-local/
    multicast/loopback/unspecified), otherwise the service hint (https, http, dns, quic, other...).
    """
    out = df.copy()
    scope_map = {ip: ip_scope(ip) for ip in out["responder_ip"].astype(str).unique()}
    out["responder_scope"] = out["responder_ip"].astype(str).map(scope_map)
    key = out["ip_proto"].astype(int) * 100000 + out["responder_port"].astype(int)
    uniq = pd.unique(key)
    svc_map = {k: service_hint(int(k // 100000), int(k % 100000)) for k in uniq}
    out["service"] = key.map(svc_map)
    out["flow_tag"] = np.where(out["responder_scope"] != "public", "local_infra", out["service"])
    return out
