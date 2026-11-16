"""
Entry-point for DDoS Detection via Federated Learning (CICIDS2017).

Usage:
    python ddos_main.py
    python ddos_main.py --data ../data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    python ddos_main.py --rounds 15 --clients 3
"""

import argparse
from ddos_detector import run_ddos_detection

def main():
    parser = argparse.ArgumentParser(
        description="DDoS Detection  Federated Learning (CICIDS2017)"
    )
    parser.add_argument(
        "--data",
        default="../data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        help="Path to CICIDS2017 DDoS CSV file"
    )
    parser.add_argument("--clients",   type=int, default=3,   help="Number of federated clients (default: 3)")
    parser.add_argument("--rounds",    type=int, default=10,  help="Number of FedAvg rounds (default: 10)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    args = parser.parse_args()

    run_ddos_detection(
        data_path=args.data,
        num_clients=args.clients,
        num_rounds=args.rounds,
        test_size=args.test_size,
    )

if __name__ == "__main__":
    main()

