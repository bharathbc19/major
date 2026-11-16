"""
FastAPI Backend for Federated Fraud & DDoS Detection Engine.
Serves web UI, handles CSV batch prediction, and manages live packet capture pipeline.
Strictly read-only for model and scaler (never calls scaler.fit()).
"""

import os
import sys
import io
import math
import time
import json
import asyncio
import tempfile
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque, defaultdict
from contextlib import asynccontextmanager

# Ensure writable matplotlib dir before any ML imports
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), ".matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Canonical 22 training features from ddos_detector.py
from ddos_detector import FEATURES as TRAINING_FEATURES

MODEL_PATHS = [
    PROJECT_ROOT / "docs" / "global_model.h5",
    PROJECT_ROOT / "models" / "global_model.h5",
]
SCALER_PATHS = [
    PROJECT_ROOT / "scaler.pkl",
    PROJECT_ROOT / "models" / "scaler.pkl",
    PROJECT_ROOT / "docs" / "scaler.pkl",
]

STATIC_DIR   = Path(__file__).resolve().parent / "static"
STATIC_INDEX = STATIC_DIR / "index.html"

GLOBAL_MODEL: Optional[tf.keras.Model] = None
GLOBAL_SCALER = None
RESOLVED_MODEL_PATH: Optional[str] = None
RESOLVED_SCALER_PATH: Optional[str] = None

INFERENCE_LOCK = threading.Lock()


def load_artifacts() -> None:
    """Load model and scaler strictly read-only once at startup."""
    global GLOBAL_MODEL, GLOBAL_SCALER, RESOLVED_MODEL_PATH, RESOLVED_SCALER_PATH
    if GLOBAL_MODEL is not None and GLOBAL_SCALER is not None:
        return
    for p in MODEL_PATHS:
        if p.exists():
            RESOLVED_MODEL_PATH = str(p)
            break
    for p in SCALER_PATHS:
        if p.exists():
            RESOLVED_SCALER_PATH = str(p)
            break
    if not RESOLVED_MODEL_PATH:
        print("[WARNING] global_model.h5 not found.")
    else:
        print(f"[OK] Loading model from {RESOLVED_MODEL_PATH}")
        GLOBAL_MODEL = tf.keras.models.load_model(RESOLVED_MODEL_PATH)
    if not RESOLVED_SCALER_PATH:
        print("[WARNING] scaler.pkl not found.")
    else:
        print(f"[OK] Loading scaler from {RESOLVED_SCALER_PATH}")
        GLOBAL_SCALER = joblib.load(RESOLVED_SCALER_PATH)


load_artifacts()


def resolve_protocol_name(proto_num: int, src_port: int, dst_port: int) -> str:
    """
    Resolve L3/L4 protocol numbers and common application protocols
    (HTTP, HTTPS, DNS, SMTP, SSH, FTP, NTP, DHCP, etc.).
    """
    ports = {int(src_port), int(dst_port)}
    if proto_num == 6:  # TCP
        if 443 in ports:
            return "HTTPS (TCP)"
        if 80 in ports or 8080 in ports:
            return "HTTP (TCP)"
        if 53 in ports:
            return "DNS (TCP)"
        if any(p in ports for p in (25, 465, 587)):
            return "SMTP (TCP)"
        if 22 in ports:
            return "SSH (TCP)"
        if 20 in ports or 21 in ports:
            return "FTP (TCP)"
        if 3389 in ports:
            return "RDP (TCP)"
        return "TCP"
    elif proto_num == 17:  # UDP
        if 53 in ports:
            return "DNS (UDP)"
        if 443 in ports:
            return "QUIC/HTTPS (UDP)"
        if 67 in ports or 68 in ports:
            return "DHCP (UDP)"
        if 123 in ports:
            return "NTP (UDP)"
        if 161 in ports or 162 in ports:
            return "SNMP (UDP)"
        return "UDP"
    elif proto_num in (1, 58):
        return "ICMP"
    elif proto_num == 2:
        return "IGMP"
    return f"IP ({proto_num})" if proto_num else "IP"


def extract_features_inline(packets) -> pd.DataFrame:
    """
    Convert a list of raw captured packets into 22 CICIDS2017 flow features.
    
    Supports ALL packet kinds: TCP, UDP, ICMP, DNS, HTTP, HTTPS, SMTP, SSH, etc.
    Applies live traffic robustness (MTU clamping, micro-burst rate flooring, fragment filtering).
    """
    try:
        from scapy.layers.inet import IP, TCP, UDP, ICMP
        from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest
    except ImportError:
        return pd.DataFrame()

    flow_buckets: dict = defaultdict(list)
    flow_init_src: dict = {}
    for pkt in packets:
        ip_layer = pkt.getlayer(IP) or pkt.getlayer(IPv6)
        if ip_layer is None:
            continue
        src_ip = getattr(ip_layer, "src", "0.0.0.0")
        dst_ip = getattr(ip_layer, "dst", "0.0.0.0")
        proto  = int(getattr(ip_layer, "proto", 0) or getattr(ip_layer, "nh", 0) or 0)
        src_port = dst_port = 0

        if pkt.haslayer(TCP):
            src_port = int(pkt[TCP].sport)
            dst_port = int(pkt[TCP].dport)
            proto = 6
        elif pkt.haslayer(UDP):
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)
            proto = 17
        elif pkt.haslayer(ICMP) or pkt.haslayer(ICMPv6EchoRequest):
            proto = 1

        raw_len = len(pkt)
        clamped_len = min(raw_len, 1500)

        key = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)])) + (proto,)
        if key not in flow_init_src:
            flow_init_src[key] = (src_ip, src_port)
        is_fwd = ((src_ip, src_port) == flow_init_src[key])

        flow_buckets[key].append({
            "ts":       float(pkt.time),
            "size":     clamped_len,
            "raw_size": raw_len,
            "is_fwd":   is_fwd,
            "src_ip":   src_ip,
            "dst_ip":   dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "proto":    proto,
        })

    rows = []
    for key, pkts in flow_buckets.items():
        if len(pkts) < 2:
            continue

        pkts.sort(key=lambda x: x["ts"])
        timestamps = [p["ts"] for p in pkts]
        sizes      = [p["size"] for p in pkts]
        fwd        = [p for p in pkts if p["is_fwd"]]
        bwd        = [p for p in pkts if not p["is_fwd"]]

        t_start = timestamps[0]
        t_end   = timestamps[-1]
        raw_duration_s = max(t_end - t_start, 0.0)
        duration_us = max(raw_duration_s * 1e6, 1.0)
        rate_duration_s = max(raw_duration_s, 0.10)

        fwd_sizes = [p["size"] for p in fwd] or [0.0]
        bwd_sizes = [p["size"] for p in bwd] or [0.0]
        all_sizes = sizes

        def iat_stats(ts_list):
            if len(ts_list) < 2:
                return 0.0, 0.0, 0.0
            iats = [(ts_list[i+1] - ts_list[i]) * 1e6 for i in range(len(ts_list)-1)]
            mean = sum(iats) / len(iats)
            std  = math.sqrt(sum((x - mean)**2 for x in iats) / len(iats)) if iats else 0.0
            return mean, std, sum(iats)

        all_iat_mean, all_iat_std, _ = iat_stats(timestamps)
        _, _, fwd_iat_total = iat_stats([p["ts"] for p in fwd])
        _, _, bwd_iat_total = iat_stats([p["ts"] for p in bwd])

        total_bytes = sum(sizes)
        total_pkts  = len(pkts)

        row = {
            "Flow Duration":                 float(duration_us),
            "Total Fwd Packets":             float(len(fwd)),
            "Total Backward Packets":        float(len(bwd)),
            "Total Length of Fwd Packets":   float(sum(fwd_sizes)),
            "Total Length of Bwd Packets":   float(sum(bwd_sizes)),
            "Fwd Packet Length Max":         float(max(fwd_sizes)),
            "Fwd Packet Length Min":         float(min(fwd_sizes)),
            "Fwd Packet Length Mean":        float(sum(fwd_sizes) / len(fwd_sizes)),
            "Bwd Packet Length Max":         float(max(bwd_sizes)),
            "Bwd Packet Length Min":         float(min(bwd_sizes)),
            "Bwd Packet Length Mean":        float(sum(bwd_sizes) / len(bwd_sizes)),
            "Flow Bytes/s":                  float(total_bytes / rate_duration_s),
            "Flow Packets/s":                float(total_pkts / rate_duration_s),
            "Flow IAT Mean":                 float(all_iat_mean),
            "Flow IAT Std":                  float(all_iat_std),
            "Fwd IAT Total":                 float(fwd_iat_total),
            "Bwd IAT Total":                 float(bwd_iat_total),
            "Packet Length Mean":            float(sum(all_sizes) / len(all_sizes)),
            "Packet Length Std":             float(math.sqrt(sum((x - sum(all_sizes)/len(all_sizes))**2 for x in all_sizes) / len(all_sizes))),
            "Average Packet Size":           float(sum(all_sizes) / total_pkts),
            "Avg Fwd Segment Size":          float(sum(fwd_sizes) / len(fwd_sizes)),
            "Avg Bwd Segment Size":          float(sum(bwd_sizes) / len(bwd_sizes)),
            # Display metadata
            "Timestamp":                     t_start,
            "Src IP":                        flow_init_src[key][0],
            "Dst IP":                        (pkts[0]["dst_ip"] if (pkts[0]["src_ip"], pkts[0]["src_port"]) == flow_init_src[key] else pkts[0]["src_ip"]),
            "Src Port":                      flow_init_src[key][1],
            "Dst Port":                      (pkts[0]["dst_port"] if (pkts[0]["src_ip"], pkts[0]["src_port"]) == flow_init_src[key] else pkts[0]["src_port"]),
            "Protocol":                      pkts[0]["proto"],
            "Protocol_Name":                 resolve_protocol_name(pkts[0]["proto"], flow_init_src[key][1], pkts[0]["dst_port"]),
            "Total Packets":                 total_pkts,
        }
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def predict_flows(df_flows: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """Run inference strictly read-only using scaler.transform()."""
    if df_flows.empty or GLOBAL_MODEL is None or GLOBAL_SCALER is None:
        df_flows = df_flows.copy()
        df_flows["probability"] = 0.0
        df_flows["verdict"] = "BENIGN"
        return df_flows

    feat_cols = [c for c in TRAINING_FEATURES if c in df_flows.columns]
    X = df_flows[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    X_scaled = GLOBAL_SCALER.transform(X).astype(np.float32)
    probs = GLOBAL_MODEL.predict(X_scaled, batch_size=512, verbose=0).flatten()

    df_out = df_flows.copy()
    df_out["probability"] = probs.astype(float)
    df_out["verdict"] = np.where(probs >= threshold, "DDoS", "BENIGN")
    return df_out


def resolve_iface_name(iface_name: Optional[str]) -> Optional[str]:
    """Resolve friendly interface aliases ('loopback', 'lo') to Npcap device names on Windows."""
    if not iface_name:
        return None
    cleaned = iface_name.strip()
    if cleaned.lower() in ("loopback", "lo", "localhost", "npcap loopback adapter"):
        try:
            from scapy.all import get_if_list
            for dev in get_if_list():
                if "loopback" in dev.lower():
                    return dev
        except Exception:
            pass
    return cleaned


def verify_capture_permissions_inline() -> tuple:
    try:
        from scapy.all import get_if_list
        ifaces = get_if_list()
        if not ifaces:
            return False, "No network interfaces found. Ensure Npcap is installed."
        return True, f"Capture ready. Interfaces: {ifaces[:3]}"
    except Exception as e:
        return False, (
            f"Packet capture not available: {e}. "
            "Ensure Npcap is installed and the terminal is running as Administrator."
        )


class LivePipelineManager:
    """Background live capture -> feature extraction -> inference loop."""

    def __init__(self):
        self.is_running  = False
        self.stop_event  = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.iface: Optional[str] = None
        self.threshold: float = 0.80
        self.last_error: Optional[str] = None
        self.total_classified = 0
        self.ddos_count  = 0
        self.benign_count = 0
        self.flow_queue: deque = deque(maxlen=500)
        self.recent_flows: deque = deque(maxlen=200)
        self.pcap_path     = str(PROJECT_ROOT / "traffic.pcap")
        self.dataset_csv   = str(PROJECT_ROOT / "live_dataset.csv")
        self.predicted_csv = str(PROJECT_ROOT / "live_dataset_predicted.csv")
        self.predictions_csv = str(PROJECT_ROOT / "predictions.csv")

    def start(self, iface: Optional[str] = None, threshold: float = 0.80):
        if self.is_running:
            return True, "Live capture pipeline is already running."
        ok, msg = verify_capture_permissions_inline()
        if not ok:
            self.last_error = msg
            return False, msg
        self.iface = iface
        self.threshold = threshold
        self.stop_event.clear()
        self.last_error = None
        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        return True, f"Live capture started (threshold: {threshold:.2f})."

    def stop(self):
        if not self.is_running:
            return True, "Live capture pipeline is not running."
        self.stop_event.set()
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        return True, "Live capture pipeline stopped."

    def reset_counters(self):
        self.total_classified = 0
        self.ddos_count = 0
        self.benign_count = 0
        self.flow_queue.clear()
        self.recent_flows.clear()

    def _run_loop(self):
        try:
            from scapy.all import sniff, PcapWriter
        except ImportError as e:
            self.last_error = f"Scapy not available: {e}"
            self.is_running = False
            return

        writer = PcapWriter(self.pcap_path, append=True, sync=True)
        try:
            while not self.stop_event.is_set():
                # Sniff all IP packets (TCP, UDP, ICMP, DNS, HTTP, HTTPS, SMTP, etc.)
                kwargs = {"count": 35, "timeout": 1.2, "store": True}
                if self.iface:
                    kwargs["iface"] = resolve_iface_name(self.iface)
                try:
                    packets = sniff(**kwargs)
                except (PermissionError, OSError) as e:
                    self.last_error = (
                        "Packet capture permission error: Administrator rights and Npcap required. "
                        f"Details: {e}"
                    )
                    self.is_running = False
                    break
                except Exception as e:
                    self.last_error = f"Sniffing error: {e}"
                    self.is_running = False
                    break

                if not packets:
                    continue
                for pkt in packets:
                    writer.write(pkt)

                df_flows = extract_features_inline(packets)
                if df_flows.empty:
                    continue

                with INFERENCE_LOCK:
                    df_pred = predict_flows(df_flows, threshold=self.threshold)

                hdr_ds = not os.path.exists(self.dataset_csv)
                df_flows.to_csv(self.dataset_csv, mode="a", header=hdr_ds, index=False)
                hdr_pred = not os.path.exists(self.predicted_csv)
                df_pred.to_csv(self.predicted_csv, mode="a", header=hdr_pred, index=False)
                hdr_p = not os.path.exists(self.predictions_csv)
                df_pred[["probability", "verdict"]].to_csv(
                    self.predictions_csv, mode="a", header=hdr_p, index=False
                )

                for _, row in df_pred.iterrows():
                    verdict = str(row["verdict"])
                    prob    = float(row["probability"])
                    self.total_classified += 1
                    if verdict == "DDoS":
                        self.ddos_count += 1
                    else:
                        self.benign_count += 1

                    ts_raw = row.get("Timestamp", time.time())
                    try:
                        ts_str = time.strftime("%H:%M:%S", time.localtime(float(ts_raw)))
                    except Exception:
                        ts_str = time.strftime("%H:%M:%S")

                    proto_disp = str(row.get("Protocol_Name", row.get("Protocol", "?")))

                    event = {
                        "time":        ts_str,
                        "src":         f"{row.get('Src IP','0.0.0.0')}:{int(row.get('Src Port',0))}",
                        "dst":         f"{row.get('Dst IP','0.0.0.0')}:{int(row.get('Dst Port',0))}",
                        "proto":       proto_disp,
                        "packets":     int(row.get("Total Packets", 1)),
                        "probability": round(prob, 4),
                        "verdict":     verdict,
                        "counters": {
                            "total":   self.total_classified,
                            "attacks": self.ddos_count,
                            "benign":  self.benign_count,
                        },
                    }
                    self.flow_queue.append(event)
                    self.recent_flows.append(event)
        finally:
            writer.close()
            self.is_running = False


pipeline_mgr = LivePipelineManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    pipeline_mgr.stop()


app = FastAPI(
    title="Federated Fraud & DDoS Detection Engine",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if not STATIC_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return HTMLResponse(content=STATIC_INDEX.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check():
    return {
        "status":              "healthy" if (GLOBAL_MODEL and GLOBAL_SCALER) else "degraded",
        "model_loaded":        GLOBAL_MODEL is not None,
        "scaler_loaded":       GLOBAL_SCALER is not None,
        "model_path":          RESOLVED_MODEL_PATH,
        "scaler_path":         RESOLVED_SCALER_PATH,
        "features_count":      len(TRAINING_FEATURES),
        "live_capture_active": pipeline_mgr.is_running,
        "live_threshold":      pipeline_mgr.threshold,
        "last_capture_error":  pipeline_mgr.last_error,
    }


@app.post("/api/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if GLOBAL_MODEL is None or GLOBAL_SCALER is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        content = file.file.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        df_raw = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if df_raw.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    col_map = {str(c).strip().lower(): c for c in df_raw.columns}
    matched_cols, missing_cols = [], []
    for feat in TRAINING_FEATURES:
        key = feat.strip().lower()
        if key in col_map:
            matched_cols.append(col_map[key])
        else:
            missing_cols.append(feat)

    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail={
                "message":        f"CSV is missing {len(missing_cols)} required feature columns.",
                "missing_columns": missing_cols,
                "found_columns":   [str(c).strip() for c in df_raw.columns],
            },
        )

    X_mat = (
        df_raw[matched_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values.astype(np.float32)
    )

    total_rows = len(X_mat)
    all_probs  = np.zeros(total_rows, dtype=np.float32)
    chunk_size = 50_000
    with INFERENCE_LOCK:
        for i in range(0, total_rows, chunk_size):
            chunk  = X_mat[i : i + chunk_size]
            scaled = GLOBAL_SCALER.transform(chunk).astype(np.float32)
            preds  = GLOBAL_MODEL.predict(scaled, batch_size=4096, verbose=0).flatten()
            all_probs[i : i + chunk_size] = preds

    verdicts  = np.where(all_probs >= 0.50, "DDoS", "BENIGN")
    attacks   = int((verdicts == "DDoS").sum())
    benign    = int((verdicts == "BENIGN").sum())
    avg_conf  = float(np.mean(all_probs)) if total_rows else 0.0

    first_500 = [
        {"row": i + 1, "probability": round(float(all_probs[i]), 4), "verdict": verdicts[i]}
        for i in range(min(500, total_rows))
    ]

    result: Dict[str, Any] = {
        "total":             total_rows,
        "attacks":           attacks,
        "benign":            benign,
        "attack_percentage": round(attacks / total_rows * 100, 2) if total_rows else 0.0,
        "avg_confidence":    round(avg_conf, 4),
        "first_500_rows":    first_500,
    }

    if "label" in col_map:
        raw_labels = df_raw[col_map["label"]].astype(str).str.strip()
        y_true = np.where(raw_labels.str.upper().isin(["DDOS", "1"]), 1, 0)
        y_pred = (all_probs >= 0.50).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        acc  = (tp + tn) / total_rows if total_rows else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        result["metrics"] = {
            "has_ground_truth": True,
            "accuracy":  round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall":    round(rec * 100, 2),
            "f1":        round(f1 * 100, 2),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }

    return result


class LiveStartRequest(BaseModel):
    iface: Optional[str] = None
    threshold: Optional[float] = 0.80


@app.post("/api/live/start")
async def start_live_capture(req: Optional[LiveStartRequest] = None):
    iface = req.iface if req else None
    threshold = float(req.threshold) if (req and req.threshold is not None) else 0.80
    success, msg = pipeline_mgr.start(iface=iface, threshold=threshold)
    if not success:
        raise HTTPException(
            status_code=403,
            detail={
                "message":    msg,
                "suggestion": "Run terminal as Administrator on Windows and ensure Npcap is installed.",
            },
        )
    return {"status": "started", "message": msg, "interface": iface or "default", "threshold": threshold}


@app.post("/api/live/stop")
async def stop_live_capture():
    success, msg = pipeline_mgr.stop()
    return {"status": "stopped", "message": msg}


@app.post("/api/live/restart")
@app.post("/api/live/reset")
async def reset_live_capture():
    """
    Stops running sniffer, clears counters and queues,
    and returns to idle state waiting for the user to click Start Capture.
    """
    pipeline_mgr.stop()
    pipeline_mgr.reset_counters()
    return {"status": "reset", "message": "Live capture session reset. Ready to start."}


@app.get("/api/live/stream")
async def live_stream():
    async def event_generator():
        last_hb = time.time()
        while True:
            sent = False
            while pipeline_mgr.flow_queue:
                flow = pipeline_mgr.flow_queue.popleft()
                yield f"data: {json.dumps(flow)}\n\n"
                sent = True
                await asyncio.sleep(0.01)
            now = time.time()
            if not sent and now - last_hb >= 2.0:
                hb = {
                    "event":      "heartbeat",
                    "active":     pipeline_mgr.is_running,
                    "counters":   {
                        "total":   pipeline_mgr.total_classified,
                        "attacks": pipeline_mgr.ddos_count,
                        "benign":  pipeline_mgr.benign_count,
                    },
                    "last_error": pipeline_mgr.last_error,
                }
                yield f"data: {json.dumps(hb)}\n\n"
                last_hb = now
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
