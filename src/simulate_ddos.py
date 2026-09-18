
from __future__ import annotations

import os
import sys
import time
import socket
import random
import signal
import argparse
from typing import Optional

# UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def get_default_target_ip() -> str:
    return "127.0.0.1"


def generate_ddos_flow(
    target_ip: str,
    target_port: int,
    packets_per_flow: int = 7,
    packet_size: int = 22,
    delay_between_packets: float = 0.01,
) -> int:
  
    payload = random._urandom(packet_size)
    bytes_sent = 0

    # New socket allocates a new ephemeral source port -> new flow in feature extractor
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for _ in range(packets_per_flow):
            sent = sock.sendto(payload, (target_ip, target_port))
            bytes_sent += sent
            if delay_between_packets > 0:
                time.sleep(delay_between_packets)
    finally:
        sock.close()

    return bytes_sent


def run_simulation(
    target_ip: str,
    target_port: int,
    num_flows: int,
    packets_per_flow: int,
    packet_size: int,
    continuous: bool = False,
    interval: float = 3.0,
    duration: Optional[int] = None,
):
    print("=" * 60)
    print("       DDoS Attack Traffic Simulator (Demonstration)")
    print("=" * 60)
    print(f"  Target IP         : {target_ip}")
    print(f"  Target Port       : {target_port}")
    print(f"  Flows per Burst   : {num_flows}")
    print(f"  Packets per Flow  : {packets_per_flow} (size: {packet_size} bytes)")
    print(f"  Mode              : {'Continuous (every ' + str(interval) + 's)' if continuous else 'Single Burst'}")
    if duration:
        print(f"  Duration Limit    : {duration} seconds")
    print("=" * 60)
    print("  [TIP] Run 'capture_live.py' in another terminal to capture this traffic.")
    print("  Press Ctrl+C at any time to stop.\n")

    start_time = time.time()
    burst_count = 0
    total_flows_sent = 0
    total_bytes_sent = 0

    stop = False

    def handle_signal(sig, frame):
        nonlocal stop
        stop = True
        print("\n\n[!] Stopping DDoS simulation...")

    import threading
    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, handle_signal)
        except (ValueError, AttributeError):
            pass

    try:
        while not stop:
            burst_count += 1
            print(f"[*] Burst #{burst_count}: Emitting {num_flows} DDoS flows...")

            burst_bytes = 0
            for i in range(num_flows):
                if stop:
                    break
                b = generate_ddos_flow(
                    target_ip=target_ip,
                    target_port=target_port,
                    packets_per_flow=packets_per_flow,
                    packet_size=packet_size,
                )
                burst_bytes += b
                total_flows_sent += 1

            total_bytes_sent += burst_bytes
            print(f"    [✓] Sent {num_flows} flows ({burst_bytes / 1024:.1f} KB)")

            if not continuous:
                break

            if duration and (time.time() - start_time) >= duration:
                print(f"[✓] Duration limit of {duration}s reached.")
                break

            # Sleep between bursts
            sleep_end = time.time() + interval
            while time.time() < sleep_end and not stop:
                time.sleep(0.2)

    except KeyboardInterrupt:
        pass

    elapsed = max(0.1, time.time() - start_time)
    print("\n" + "-" * 60)
    print("  Simulation Summary:")
    print(f"    Total Bursts Sent  : {burst_count}")
    print(f"    Total Flows Sent   : {total_flows_sent}")
    print(f"    Total Data Emitted : {total_bytes_sent / 1024:.2f} KB")
    print(f"    Elapsed Time       : {elapsed:.1f} seconds")
    print(f"    Rate               : {total_flows_sent / elapsed:.1f} flows/sec")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate DDoS attack traffic for live demonstration."
    )
    default_ip = get_default_target_ip()

    parser.add_argument(
        "--target",
        type=str,
        default=default_ip,
        help=f"Target IP address (default: {default_ip} - broadcast address)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help="Target port (default: 9999)",
    )
    parser.add_argument(
        "--flows",
        type=int,
        default=30,
        help="Number of DDoS flows to generate per burst (default: 30)",
    )
    parser.add_argument(
        "--packets",
        type=int,
        default=7,
        help="Packets per flow burst (default: 7 - optimal for CICIDS2017 detection)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=22,
        help="Packet payload size in bytes (default: 22 — yields 64B wire size for 98%% DDoS detection)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously in background until stopped",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Interval in seconds between bursts in continuous mode (default: 3.0)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Maximum duration in seconds to run (optional)",
    )

    args = parser.parse_args()

    run_simulation(
        target_ip=args.target,
        target_port=args.port,
        num_flows=args.flows,
        packets_per_flow=args.packets,
        packet_size=args.size,
        continuous=args.continuous,
        interval=args.interval,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
