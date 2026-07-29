"""Fase 2 — Explicabilidad del scoring con SHAP.

Salidas:
- reports/figures/shap_summary.png      — beeswarm global (qué mueve el riesgo en la cartera)
- reports/figures/shap_importance.png   — importancia media |SHAP| por variable
- reports/figures/shap_waterfall.png    — explicación local del cliente de mayor riesgo
- reports/reason_codes_examples.json    — reason codes de 3 clientes (alto / medio / bajo riesgo)

Decisiones:
- TreeExplainer sobre el XGBoost entrenado en fase 1 (exacto para árboles, no aproximado).
- Los reason codes son DETERMINISTAS (plantillas sobre los SHAP values): una explicación
  de denegación debe ser reproducible ante un regulador, no generada por un LLM.
- Los atributos protegidos (SEX, AGE, MARRIAGE) NUNCA aparecen como reason code,
  aunque el modelo los use: práctica estándar de scoring en la UE.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "credit_default.csv"
MODEL = ROOT / "models" / "xgb.json"
FIGURES = ROOT / "reports" / "figures"
SEED = 42

from reasons import reason_codes  # noqa: E402 (módulo compartido con la API)


def main() -> None:
    df = pd.read_csv(DATA)
    X = df.drop(columns=["default"])
    y = df["default"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    model = XGBClassifier()
    model.load_model(MODEL)
    scores = model.predict_proba(X_test)[:, 1]

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)

    FIGURES.mkdir(parents=True, exist_ok=True)

    # Global: beeswarm
    plt.figure()
    shap.summary_plot(explanation, X_test, show=False, max_display=15)
    plt.title("Qué mueve el riesgo de impago (SHAP, test 20%)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Global: importancia media
    plt.figure()
    shap.summary_plot(explanation, X_test, plot_type="bar", show=False, max_display=15)
    plt.title("Importancia media |SHAP|")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Local: waterfall del cliente con mayor probabilidad de impago
    worst = int(np.argmax(scores))
    plt.figure()
    shap.plots.waterfall(explanation[worst], max_display=12, show=False)
    plt.title(f"Cliente de mayor riesgo — p(default) = {scores[worst]:.2f}")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Reason codes de 3 clientes representativos
    features = list(X_test.columns)
    shap_vals = explanation.values
    picks = {
        "alto_riesgo": worst,
        "riesgo_medio": int(np.argsort(scores)[len(scores) // 2]),
        "bajo_riesgo": int(np.argmin(scores)),
    }
    medians = X_test.median()
    examples = {}
    for name, i in picks.items():
        examples[name] = {
            "p_default": round(float(scores[i]), 4),
            "default_real": int(y_test.iloc[i]),
            "reason_codes": reason_codes(X_test.iloc[i], shap_vals[i], features, medians),
        }
    out = ROOT / "reports" / "reason_codes_examples.json"
    out.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(examples, indent=2, ensure_ascii=False))
    print(f"\nfiguras en {FIGURES}")


if __name__ == "__main__":
    main()
