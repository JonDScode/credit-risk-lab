"""Fase 4 — Variante Deep Learning: MLP tabular en PyTorch.

Mismo split y métricas que el baseline (train.py) para una comparativa honesta.
Arquitectura: 23 → 128 → 64 → 1 con ReLU y dropout; BCEWithLogitsLoss con
pos_weight para el desbalance (~22% de impagos); early stopping sobre AUC
de validación.

Añade la fila "pytorch_mlp" a reports/metrics.json y guarda models/mlp.pt.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from train import evaluate  # mismas métricas que el baseline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "credit_default.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_model(n_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    )


def auc_score(model: nn.Module, X: torch.Tensor, y: np.ndarray) -> tuple[float, np.ndarray]:
    from sklearn.metrics import roc_auc_score
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X)).squeeze(1).numpy()
    return roc_auc_score(y, scores), scores


def main() -> None:
    df = pd.read_csv(DATA)
    X = df.drop(columns=["default"])
    y = df["default"].to_numpy()

    # Mismo test del 20% que train.py; del 80% restante, 10% para validación
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    Xt_train, Xt_val, Xt_test = (to_t(scaler.transform(s)) for s in (X_train, X_val, X_test))
    yt_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = build_model(X.shape[1])
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    dataset = torch.utils.data.TensorDataset(Xt_train, yt_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    best_auc, best_state, patience, bad_epochs = 0.0, None, 15, 0
    for epoch in range(200):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_auc, _ = auc_score(model, Xt_val, y_val)
        if val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping en epoch {epoch} (mejor val AUC {best_auc:.4f})")
                break

    model.load_state_dict(best_state)
    _, test_scores = auc_score(model, Xt_test, y_test)
    result = evaluate("pytorch_mlp", y_test, test_scores)

    metrics_path = REPORTS / "metrics.json"
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    results = [r for r in results if r["model"] != "pytorch_mlp"] + [result]
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    MODELS.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist()}, MODELS / "mlp.pt")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
