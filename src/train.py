"""Entrena el baseline de scoring y guarda métricas + figuras.

Modelos:
- Regresión logística (referencia interpretable, estándar del sector)
- XGBoost (baseline serio)

Métricas de la casa (banca):
- ROC AUC        — discriminación global
- PR AUC         — comportamiento en la clase minoritaria (impagos)
- KS statistic   — separación entre buenos y malos (métrica clásica de scoring)
- Brier score    — calibración de las probabilidades

Salidas: reports/metrics.json, reports/figures/roc.png, models/xgb.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "credit_default.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 42


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(max(tpr - fpr))


def evaluate(name, y_true, y_score) -> dict:
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "brier": round(brier_score_loss(y_true, y_score), 4),
    }


def main() -> None:
    df = pd.read_csv(DATA)
    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Referencia: logística con escalado
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    logit.fit(X_train, y_train)
    logit_scores = logit.predict_proba(X_test)[:, 1]

    # Baseline: XGBoost
    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=SEED,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_scores = xgb.predict_proba(X_test)[:, 1]

    results = [
        evaluate("logistic_regression", y_test, logit_scores),
        evaluate("xgboost", y_test, xgb_scores),
    ]

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Curva ROC comparada
    fig, ax = plt.subplots(figsize=(6, 5))
    for scores, label in [(logit_scores, "Logistic"), (xgb_scores, "XGBoost")]:
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(y_test, scores):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("ROC — scoring de impago (test 20%)")
    ax.legend()
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "roc.png", dpi=150)

    MODELS.mkdir(exist_ok=True)
    xgb.save_model(MODELS / "xgb.json")

    print(json.dumps(results, indent=2))
    print(f"\nfiguras: {REPORTS / 'figures' / 'roc.png'}")
    print(f"modelo:  {MODELS / 'xgb.json'}")


if __name__ == "__main__":
    main()
