

from __future__ import annotations

import os
import sys
import math
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── UTF-8 stdout ───────────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure src/ is on path ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import (  # noqa: E402
    FEATURES,
    PCAP_PATH,
    LIVE_DATASET_CSV,
    get_logger,
    validate_features,
    ensure_dir,
    safe_div,
)

log = get_logger("feature_extractor")

# ── Scapy import ───────────────────────────────────────────────────────────────
try:
    from scapy.all import PcapReader
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6
except ImportError as exc:
    log.error("Scapy is not installed.  Run: pip install scapy")
    raise SystemExit(1) from exc

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FLOW_IDLE_TIMEOUT   = 120.0   # seconds
FLOW_ACTIVE_TIMEOUT = 600.0   # seconds
MIN_PACKETS         = 2       # flows with fewer packets are discarded
LABEL_UNKNOWN       = "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Internal data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _PktRecord:
    timestamp: float
    length: int


@dataclass
class _Flow:
    key:      tuple
    start_ts: float
    last_ts:  float
    init_src: tuple = field(default_factory=tuple)

    fwd_pkts: List[_PktRecord] = field(default_factory=list)
    bwd_pkts: List[_PktRecord] = field(default_factory=list)

    def add_packet(self, ts: float, length: int, is_forward: bool) -> None:
        rec = _PktRecord(ts, length)
        if is_forward:
            self.fwd_pkts.append(rec)
        else:
            self.bwd_pkts.append(rec)
        if ts > self.last_ts:
            self.last_ts = ts

    @property
    def all_pkts(self) -> List[_PktRecord]:
        combined = self.fwd_pkts + self.bwd_pkts
        combined.sort(key=lambda p: p.timestamp)
        return combined

    @property
    def total_pkts(self) -> int:
        return len(self.fwd_pkts) + len(self.bwd_pkts)


# ─────────────────────────────────────────────────────────────────────────────
# Feature calculation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iat_stats(records: List[_PktRecord]) -> Tuple[float, float, float]:
    """
    Return (total_iat_us, mean_iat_us, std_iat_us) from a chronologically
    sorted list of packet records.  All values in microseconds.
    """
    if len(records) < 2:
        return 0.0, 0.0, 0.0
    iats = [
        (records[i].timestamp - records[i - 1].timestamp) * 1e6
        for i in range(1, len(records))
    ]
    total    = sum(iats)
    mean     = total / len(iats)
    variance = sum((x - mean) ** 2 for x in iats) / len(iats)
    return total, mean, math.sqrt(variance)


def _length_stats(records: List[_PktRecord]) -> Tuple[float, float, float, float, float]:
    """
    Return (max, min, mean, total_bytes, avg_segment_size) for a list of
    packet records.  avg_segment_size = mean (matches CICFlowMeter).
    """
    if not records:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    lengths = [r.length for r in records]
    total   = float(sum(lengths))
    mx      = float(max(lengths))
    mn      = float(min(lengths))
    mean    = total / len(lengths)
    return mx, mn, mean, total, mean   # seg_size == mean


# ─────────────────────────────────────────────────────────────────────────────
# Per-flow feature extraction (22 features)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(flow: _Flow) -> Dict[str, float]:
    """
    Extract the 22 CICIDS2017 features from *flow*.

    • Timing values are in microseconds (matches CICFlowMeter).
    • Lengths are in bytes.
    • Returns a dict whose keys are exactly the FEATURES list.
    """
    all_pkts = flow.all_pkts
    n_total  = len(all_pkts)

    # Duration
    flow_duration_us = max((flow.last_ts - flow.start_ts) * 1e6, 0.0)

    # Flow-level IAT (all packets, direction-agnostic)
    _, flow_iat_mean, flow_iat_std = _iat_stats(all_pkts)

    # Per-direction sorted lists
    fwd_sorted = sorted(flow.fwd_pkts, key=lambda p: p.timestamp)
    bwd_sorted = sorted(flow.bwd_pkts, key=lambda p: p.timestamp)

    fwd_iat_total, _, _ = _iat_stats(fwd_sorted)
    bwd_iat_total, _, _ = _iat_stats(bwd_sorted)

    n_fwd = len(flow.fwd_pkts)
    n_bwd = len(flow.bwd_pkts)

    fwd_max, fwd_min, fwd_mean, fwd_total, fwd_seg = _length_stats(fwd_sorted)
    bwd_max, bwd_min, bwd_mean, bwd_total, bwd_seg = _length_stats(bwd_sorted)

    all_lengths   = [r.length for r in all_pkts]
    pkt_len_mean  = float(np.mean(all_lengths)) if all_lengths else 0.0
    pkt_len_std   = float(np.std(all_lengths))  if all_lengths else 0.0
    total_payload = sum(all_lengths)
    avg_pkt_size  = safe_div(float(total_payload), float(n_total))

    duration_s     = flow_duration_us / 1e6 if flow_duration_us > 0 else 1e-6
    flow_bytes_s   = safe_div(float(total_payload), duration_s)
    flow_packets_s = safe_div(float(n_total),        duration_s)

    return {
        "Flow Duration":                float(flow_duration_us),
        "Total Fwd Packets":            float(n_fwd),
        "Total Backward Packets":       float(n_bwd),
        "Total Length of Fwd Packets":  float(fwd_total),
        "Total Length of Bwd Packets":  float(bwd_total),
        "Fwd Packet Length Max":        float(fwd_max),
        "Fwd Packet Length Min":        float(fwd_min),
        "Fwd Packet Length Mean":       float(fwd_mean),
        "Bwd Packet Length Max":        float(bwd_max),
        "Bwd Packet Length Min":        float(bwd_min),
        "Bwd Packet Length Mean":       float(bwd_mean),
        "Flow Bytes/s":                 float(flow_bytes_s),
        "Flow Packets/s":               float(flow_packets_s),
        "Flow IAT Mean":                float(flow_iat_mean),
        "Flow IAT Std":                 float(flow_iat_std),
        "Fwd IAT Total":                float(fwd_iat_total),
        "Bwd IAT Total":                float(bwd_iat_total),
        "Packet Length Mean":           float(pkt_len_mean),
        "Packet Length Std":            float(pkt_len_std),
        "Average Packet Size":          float(avg_pkt_size),
        "Avg Fwd Segment Size":         float(fwd_seg),
        "Avg Bwd Segment Size":         float(bwd_seg),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Packet parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_packet(pkt) -> Tuple[Optional[str], Optional[str], int, int, int, float, int]:
    """
    Parse a Scapy packet.
    Returns (src_ip, dst_ip, src_port, dst_port, proto, timestamp, length).
    Returns (None, None, 0, 0, 0, 0.0, 0) for malformed/unsupported packets.
    """
    try:
        ts = float(pkt.time)

        if pkt.haslayer(IP):
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto  = int(pkt[IP].proto)
        elif pkt.haslayer(IPv6):
            src_ip = pkt[IPv6].src
            dst_ip = pkt[IPv6].dst
            proto  = int(pkt[IPv6].nh)
        else:
            return None, None, 0, 0, 0, 0.0, 0

        src_port, dst_port = 0, 0
        if pkt.haslayer(TCP):
            src_port = int(pkt[TCP].sport)
            dst_port = int(pkt[TCP].dport)
        elif pkt.haslayer(UDP):
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)

        return src_ip, dst_ip, src_port, dst_port, proto, ts, len(pkt)

    except Exception:
        return None, None, 0, 0, 0, 0.0, 0


def _flow_key(src_ip: str, dst_ip: str,
              src_port: int, dst_port: int,
              proto: int) -> tuple:
    """Canonical bidirectional flow key (A→B == B→A)."""
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a > b:
        a, b = b, a
    return (*a, *b, proto)


# ─────────────────────────────────────────────────────────────────────────────
# Flow table
# ─────────────────────────────────────────────────────────────────────────────

class FlowTable:
    """Tracks active flows and emits completed ones on timeout."""

    def __init__(self,
                 idle_timeout: float = FLOW_IDLE_TIMEOUT,
                 active_timeout: float = FLOW_ACTIVE_TIMEOUT) -> None:
        self._flows: Dict[tuple, _Flow] = {}
        self._idle_timeout   = idle_timeout
        self._active_timeout = active_timeout

    def add_packet(self, src_ip: str, dst_ip: str,
                   src_port: int, dst_port: int,
                   proto: int, ts: float, length: int) -> List[_Flow]:
        """Add packet to the appropriate flow; return any newly completed flows."""
        key = _flow_key(src_ip, dst_ip, src_port, dst_port, proto)
        completed: List[_Flow] = []

        if key in self._flows:
            flow = self._flows[key]
            gap  = ts - flow.last_ts

            if gap > self._idle_timeout:
                completed.append(self._flows.pop(key))
                flow = _Flow(key=key, start_ts=ts, last_ts=ts, init_src=(src_ip, src_port))
                self._flows[key] = flow
            elif (ts - flow.start_ts) > self._active_timeout:
                completed.append(self._flows.pop(key))
                flow = _Flow(key=key, start_ts=ts, last_ts=ts, init_src=(src_ip, src_port))
                self._flows[key] = flow

            is_forward = ((src_ip, src_port) == flow.init_src)
            flow.add_packet(ts, length, is_forward)
        else:
            flow = _Flow(key=key, start_ts=ts, last_ts=ts, init_src=(src_ip, src_port))
            is_forward = True
            flow.add_packet(ts, length, is_forward)
            self._flows[key] = flow

        return completed

    def flush_all(self) -> List[_Flow]:
        flows = list(self._flows.values())
        self._flows.clear()
        return flows

    def __len__(self) -> int:
        return len(self._flows)


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_pcap(
    pcap_path: str = PCAP_PATH,
    out_csv: str   = LIVE_DATASET_CSV,
    idle_timeout: float  = FLOW_IDLE_TIMEOUT,
    active_timeout: float = FLOW_ACTIVE_TIMEOUT,
    min_packets: int = MIN_PACKETS,
) -> pd.DataFrame:
    """
    Read *pcap_path* packet-by-packet, build flows, extract the 22 CICIDS2017
    features, and write a CICIDS2017-compatible *out_csv*.

    Output CSV structure:
        Flow Duration, Total Fwd Packets, ..., Avg Bwd Segment Size, Label
        (22 feature columns in canonical order)  +  Label = "Unknown"

    Args:
        pcap_path:      Input .pcap file.
        out_csv:        Destination CSV path (default: live_dataset.csv).
        idle_timeout:   Flow idle timeout in seconds.
        active_timeout: Maximum flow active duration in seconds.
        min_packets:    Flows with fewer packets than this are discarded.

    Returns:
        DataFrame with the 22 feature columns only (Label excluded).
        Empty DataFrame if no flows were extracted.
    """
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    log.info("Reading: %s", pcap_path)
    ensure_dir(out_csv)

    table        = FlowTable(idle_timeout=idle_timeout, active_timeout=active_timeout)
    raw_records: List[Dict[str, Any]] = []
    flow_id      = 0
    pkt_count    = 0
    skipped      = 0

    def _record_flow(flow: _Flow) -> None:
        nonlocal flow_id, skipped
        if flow.total_pkts < min_packets:
            skipped += 1
            return
        feats = extract_features(flow)
        raw_records.append(feats)
        flow_id += 1

    try:
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                pkt_count += 1
                src_ip, dst_ip, src_port, dst_port, proto, ts, length = _parse_packet(pkt)
                if src_ip is None:
                    continue

                for f in table.add_packet(src_ip, dst_ip, src_port, dst_port, proto, ts, length):
                    _record_flow(f)

                if pkt_count % 10_000 == 0:
                    log.info(
                        "Processed %d pkts | Active flows: %d | Completed: %d",
                        pkt_count, len(table), flow_id,
                    )
    except Exception as exc:
        log.error("Error reading PCAP: %s", exc)
        raise

    for f in table.flush_all():
        _record_flow(f)

    log.info(
        "Done.  Packets: %d | Flows extracted: %d | Skipped (<% d pkts): %d",
        pkt_count, flow_id, min_packets, skipped,
    )

    if not raw_records:
        log.warning("No usable flows — writing empty live_dataset.csv")
        df_empty = pd.DataFrame(columns=FEATURES + ["Label"])
        df_empty.to_csv(out_csv, index=False)
        return pd.DataFrame(columns=FEATURES)

    # ── Build feature DataFrame ────────────────────────────────────────────────
    df_feat = pd.DataFrame(raw_records)

    # Validate: enforce canonical order, fill missing, sanitise ±inf / NaN
    df_feat = validate_features(df_feat[FEATURES].copy())   # → float32, 22 cols

    # ── Build CICIDS2017-style output ──────────────────────────────────────────
    # Cast to float64 to match the training CSV dtypes
    df_out = df_feat.astype("float64").copy()

    # Add Label column — "Unknown" until inference replaces it
    df_out["Label"] = LABEL_UNKNOWN

    # Columns: [22 features in order] + [Label]
    df_out.to_csv(out_csv, index=False)
    log.info(
        "live_dataset.csv saved → %s  (%d rows × %d cols: 22 features + Label)",
        out_csv, len(df_out), len(df_out.columns),
    )

    # Return 22-feature frame (float32) for pipeline chaining to live_predict
    return df_feat


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CICIDS2017-style features from .pcap → live_dataset.csv"
    )
    parser.add_argument("--pcap",        default=PCAP_PATH,       help="Input .pcap file")
    parser.add_argument("--out",         default=LIVE_DATASET_CSV, help="Output CSV path")
    parser.add_argument("--timeout",     type=float, default=FLOW_IDLE_TIMEOUT,   metavar="S", help="Flow idle timeout (s)")
    parser.add_argument("--min-packets", type=int,   default=MIN_PACKETS,         metavar="N", help="Min packets per flow")
    args = parser.parse_args()

    df = extract_from_pcap(
        pcap_path=args.pcap,
        out_csv=args.out,
        idle_timeout=args.timeout,
        min_packets=args.min_packets,
    )
    print(f"\n[✓] Extracted {len(df)} flows → {args.out}")


if __name__ == "__main__":
    main()
