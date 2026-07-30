"""Fase 5 — Scoring sobre Home Credit (307k solicitudes, features multi-tabla).

Mismo protocolo que el resto del proyecto: split estratificado 80/20, XGBoost
con early stopping y la misma batería de métricas (AUC, PR-AUC, KS, Brier).

Referencia externa: en la competición de Kaggle, un modelo single ~0.75-0.78
de AUC privado; los ganadores (ensembles masivos) ~0.80.

Salidas: reports/metrics_homecredit.json, reports/figures/roc_homecredit.png,
         models/xgb_homecredit.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from train import evaluate

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "homecredit_features.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 42


def main() -> None:
    df = pd.read_csv(DATA)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=20,
        scale_pos_weight=1,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=200)
    scores = model.predict_proba(X_test)[:, 1]

    result = evaluate("xgboost_homecredit", y_test, scores)
    (REPORTS / "metrics_homecredit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fpr, tpr, _ = roc_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"XGBoost (AUC {roc_auc_score(y_test, scores):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("ROC — Home Credit (61k solicitudes de test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "roc_homecredit.png", dpi=150)

    MODELS.mkdir(exist_ok=True)
    model.save_model(MODELS / "xgb_homecredit.json")

    # Top features por ganancia — sanity check de negocio
    imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(15)
    print(json.dumps(result, indent=2))
    print("\nTop features:")
    print(imp.to_string())


if __name__ == "__main__":
    main()
