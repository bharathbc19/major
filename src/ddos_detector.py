"""
DDoS Detection using Federated Learning on CICIDS2017 Dataset.
Simulates FedAvg across 3 client nodes to detect DDoS vs BENIGN traffic.
Run via: python ddos_main.py
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")           # no display needed  saves to file
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

#  Experiment flag 
# Set True to distribute data unevenly (DDoS-heavy / balanced / BENIGN-heavy)
NON_IID = True

# Output paths
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
OUT_DIR          = os.path.abspath(os.path.join(BASE_DIR, "..", "docs"))
CONVERGENCE_PLOT = os.path.join(OUT_DIR, "convergence_plot.png")
ROC_PLOT         = os.path.join(OUT_DIR, "roc_curve.png")
MODEL_PATH       = os.path.join(OUT_DIR, "global_model.h5")

#  Features 
FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Total", "Bwd IAT Total", "Packet Length Mean", "Packet Length Std",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size"
]

LABEL_COL  = "Label"
DDOS_LABEL = "DDoS"


#  Data Loading 
def load_cicids2017(file_path):
    """Load CICIDS2017 CSV, strip whitespace from column names and labels."""
    print(f"[OK] Loading {file_path} ...")
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df[LABEL_COL] = df[LABEL_COL].str.strip()
    print(f"[OK] Loaded {len(df):,} rows - {df.shape[1]} columns")
    return df


#  Labelling 
def label_ddos(df):
    """Binary label: 1 = DDoS, 0 = BENIGN. Drop any other labels."""
    df = df[df[LABEL_COL].isin([DDOS_LABEL, "BENIGN"])].copy()
    df["is_ddos"] = (df[LABEL_COL] == DDOS_LABEL).astype(int)
    ddos_n = int(df["is_ddos"].sum())
    norm_n = int((df["is_ddos"] == 0).sum())
    print(f"[OK] Labeled    DDoS: {ddos_n:,}  |  BENIGN: {norm_n:,}  "
          f"|  DDoS rate: {ddos_n / len(df) * 100:.1f}%")
    return df


#  Preprocessing 
def preprocess(df, scaler=None, fit_scaler=True):
    """Select features, replace inf/nan, scale."""
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    y = df["is_ddos"].values.astype(np.float32)
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)
    return X, y, scaler


#  IID Split 
def split_clients_iid(X, y, num_clients=3):
    """Split data equally across clients (IID)."""
    n, size = len(X), len(X) // num_clients
    clients = []
    for i in range(num_clients):
        start = i * size
        end = (i + 1) * size if i < num_clients - 1 else n
        clients.append((X[start:end], y[start:end]))
    return clients


#  Non-IID Split 
def split_clients_non_iid(X, y, num_clients=3):
    """
    Non-IID distribution for 3 clients:
      Client 1  80% DDoS-heavy
      Client 2  balanced (~50/50)
      Client 3  80% BENIGN-heavy
    """
    ddos_idx  = np.where(y == 1)[0]
    benign_idx = np.where(y == 0)[0]
    np.random.seed(42)
    np.random.shuffle(ddos_idx)
    np.random.shuffle(benign_idx)

    n_ddos, n_benign = len(ddos_idx), len(benign_idx)
    shard = min(n_ddos, n_benign) // num_clients

    # Client 1: 80% DDoS
    c1_ddos   = ddos_idx[:int(shard * 0.8)]
    c1_benign = benign_idx[:int(shard * 0.2)]
    c1_idx    = np.concatenate([c1_ddos, c1_benign])

    # Client 2: ~50/50 balanced
    c2_ddos   = ddos_idx[int(shard * 0.8): int(shard * 0.8) + shard // 2]
    c2_benign = benign_idx[int(shard * 0.2): int(shard * 0.2) + shard // 2]
    c2_idx    = np.concatenate([c2_ddos, c2_benign])

    # Client 3: 80% BENIGN
    c3_ddos   = ddos_idx[int(shard * 0.8) + shard // 2: int(shard * 0.8) + shard // 2 + int(shard * 0.2)]
    c3_benign = benign_idx[int(shard * 0.2) + shard // 2: int(shard * 0.2) + shard // 2 + int(shard * 0.8)]
    c3_idx    = np.concatenate([c3_ddos, c3_benign])

    clients = []
    for idx in [c1_idx, c2_idx, c3_idx]:
        np.random.shuffle(idx)
        clients.append((X[idx], y[idx]))
    return clients


#  Model 
def build_model(input_dim):
    """Build model using explicit Input() layer (avoids Keras warning)."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),          # Fix #1: explicit Input layer
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


#  FedAvg Helpers 
def local_train(weights, X, y, epochs=3, batch_size=256):
    model = build_model(X.shape[1])
    model.set_weights(weights)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model.get_weights(), len(X)


def federated_average(all_weights, all_sizes):
    total = sum(all_sizes)
    return [
        sum(w * (s / total) for w, s in zip(layers, all_sizes))
        for layers in zip(*all_weights)
    ]


#  Metrics 
def evaluate_model(model, X, y):
    loss, acc = model.evaluate(X, y, verbose=0)
    y_pred = (model.predict(X, verbose=0) > 0.5).astype(int).flatten()
    tp = int(((y_pred == 1) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return dict(loss=loss, acc=acc, precision=precision, recall=recall,
                f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


#  Convergence Plot 
def plot_convergence(round_losses, round_accs, save_path):
    rounds = list(range(1, len(round_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(rounds, round_losses, "o-", color="#e74c3c", linewidth=2, label="Loss")
    ax1.set_title("Federated Training  Loss per Round", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Round"); ax1.set_ylabel("Loss")
    ax1.set_xticks(rounds); ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(rounds, [a * 100 for a in round_accs], "o-", color="#2ecc71", linewidth=2, label="Accuracy")
    ax2.set_title("Federated Training  Accuracy per Round", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Round"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(rounds); ax2.grid(True, alpha=0.3); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Convergence plot saved  {save_path}")


#  ROC Curve 
def plot_roc(model, X_test, y_test, save_path):
    y_scores = model.predict(X_test, verbose=0).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#3498db", lw=2,
             label=f"Global Model  (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve  DDoS Detection (Global Model)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] ROC curve saved  {save_path}  (AUC = {roc_auc:.4f})")
    return roc_auc


#  Main Entry 
def run_ddos_detection(data_path, num_clients=3, num_rounds=10,
                       test_size=0.2, non_iid=NON_IID):

    print("\n" + "=" * 60)
    print("   DDoS Detection  Federated Learning (CICIDS2017)")
    mode_tag = "Non-IID" if non_iid else "IID"
    print(f"   Distribution mode: {mode_tag}")
    print("=" * 60)

    # Load & label
    df = load_cicids2017(data_path)
    df = label_ddos(df)

    # Preprocess
    X, y, scaler = preprocess(df, fit_scaler=True)

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[OK] Split    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Distribute across clients
    if non_iid:
        clients = split_clients_non_iid(X_train, y_train, num_clients)
        print(f"\n[OK] Non-IID data distributed across {num_clients} federated clients:")
    else:
        clients = split_clients_iid(X_train, y_train, num_clients)
        print(f"\n[OK] IID data distributed across {num_clients} federated clients:")

    for i, (cx, cy) in enumerate(clients):
        print(f"    Client {i+1}: {len(cx):,} samples | DDoS rate: {cy.mean()*100:.1f}%")

    # FedAvg training loop
    global_model   = build_model(X_train.shape[1])
    global_weights = global_model.get_weights()

    print(f"\n{''*60}")
    print(f"  Starting FedAvg  {num_rounds} communication rounds  [{mode_tag}]")
    print(f"{''*60}")

    round_losses, round_accs = [], []
    last_local_weights = []         # keep last round's per-client weights

    for rnd in range(1, num_rounds + 1):
        local_wts, local_sizes = [], []
        for cx, cy in clients:
            lw, ls = local_train(global_weights, cx, cy)
            local_wts.append(lw)
            local_sizes.append(ls)
        global_weights = federated_average(local_wts, local_sizes)
        last_local_weights = local_wts          # save for comparison

        eval_m = build_model(X_test.shape[1])
        eval_m.set_weights(global_weights)
        loss, acc = eval_m.evaluate(X_test, y_test, verbose=0)
        round_losses.append(loss)
        round_accs.append(acc)
        print(f"  Round {rnd:2d}/{num_rounds}  Loss: {loss:.4f}  Accuracy: {acc*100:.2f}%")

    #  Final global model 
    final_model = build_model(X_test.shape[1])
    final_model.set_weights(global_weights)
    m = evaluate_model(final_model, X_test, y_test)

    print(f"\n{''*60}")
    print("  Final Global Model Evaluation  (Test Set)")
    print(f"{''*60}")
    print(f"  Loss       : {m['loss']:.4f}")
    print(f"  Accuracy   : {m['acc']*100:.2f}%")
    print(f"  Precision  : {m['precision']*100:.2f}%")
    print(f"  Recall     : {m['recall']*100:.2f}%")
    print(f"  F1-Score   : {m['f1']*100:.2f}%")
    print(f"\n  True Positives  (DDoS caught)       : {m['tp']:,}")
    print(f"  True Negatives  (BENIGN correct)    : {m['tn']:,}")
    print(f"  False Positives (False alarms)      : {m['fp']:,}")
    print(f"  False Negatives (Missed DDoS)       : {m['fn']:,}")

    #  Local vs Global comparison 
    print(f"\n{''*60}")
    print("  Local vs Global Model Comparison  (Test Set Accuracy)")
    print(f"{''*60}")
    print(f"  {'Model':<12} {'Accuracy':>10}")
    print(f"  {''*22}")
    for i, lw in enumerate(last_local_weights):
        lm = build_model(X_test.shape[1])
        lm.set_weights(lw)
        _, local_acc = lm.evaluate(X_test, y_test, verbose=0)
        print(f"  {'Client ' + str(i+1):<12} {local_acc*100:>9.2f}%")
    print(f"  {''*22}")
    print(f"  {'Global':<12} {m['acc']*100:>9.2f}%")

    #  Save global model 
    os.makedirs(OUT_DIR, exist_ok=True)
    final_model.save(MODEL_PATH)
    print(f"\n[OK] Global model saved  {MODEL_PATH}")

    #  Convergence plot 
    plot_convergence(round_losses, round_accs, CONVERGENCE_PLOT)

    #  ROC curve + AUC 
    roc_auc = plot_roc(final_model, X_test, y_test, ROC_PLOT)

    print(f"\n  AUC Score  : {roc_auc:.4f}")
    print(f"\n{'='*60}")
    print("  DDoS Detection Complete!")
    print(f"{'='*60}\n")

