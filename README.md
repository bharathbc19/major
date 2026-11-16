# Fraud Detection Federated

Federated learning-based fraud detection system enabling secure, privacy-preserving model training across financial institutions without sharing raw data. Supports real-time and batch processing, anomaly detection, and modular integration with external APIs.

## Project Structure

- `fraud-detection-federated/`
  - `data/`: Contains synthetic dataset `synthetic_data.csv`.
  - `docs/`: Documentation and figures.
  - `models/`: Saved models after federated training.
  - `notebooks/`: Jupyter notebooks.
  - `src/`: Python source code files.

## Installation

### 1. Prerequisites
- Python 3.8+ 

### 2. Install Dependencies
Run the following from the root directory (`d:\major\fraud-detection-federated`):

```sh
pip install -r requirements.txt
```

*(This will install pandas, matplotlib, seaborn, tensorflow, scikit-learn)*

> **Note for Windows Users**: The `tensorflow-federated` package does not support Windows natively via pip. Therefore, `requirements.txt` only includes the core packages required to run basic models and analysis (`main.py`). To run the federated simulation (`federated_train.py`), you must install `tensorflow-federated` on a Linux system or within **WSL** (Windows Subsystem for Linux).

## How to Run

Navigate precisely to the `src` folder before running scripts, as the scripts use relative paths (e.g., `../data/synthetic_data.csv`).

```sh
cd fraud-detection-federated/src
```

### 1. Basic Data Analysis and Visualization
To view basic summary statistics of the dataset:
```sh
python main.py --summary
```

To see fraud statistics:
```sh
python main.py --fraud
```

To generate and save charts (creates PNGs in `docs/`):
```sh
python main.py --charts
```

To view risk analysis based on locations and types:
```sh
python main.py --risks
```

### 2. Federated Training Simulation
To start the federated training simulation across multiple clients:
```sh
python federated_train.py
```
This sets up simulated clients using `tensorflow-federated` and trains a global model iteratively.