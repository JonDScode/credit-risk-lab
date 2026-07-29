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

# Nunca aparecen como reason code: protegidos por normativa (SEX, AGE, MARRIAGE)
# o éticamente cuestionables como motivo de denegación (EDUCATION).
EXCLUDED_FROM_REASONS = {"SEX", "AGE", "MARRIAGE", "EDUCATION"}

LABELS = {
    "LIMIT_BAL": "Límite de crédito concedido",
    "EDUCATION": "Nivel educativo",
    "PAY_1": "Estado de pago del último mes",
    "PAY_2": "Estado de pago hace 2 meses",
    "PAY_3": "Estado de pago hace 3 meses",
    "PAY_4": "Estado de pago hace 4 meses",
    "PAY_5": "Estado de pago hace 5 meses",
    "PAY_6": "Estado de pago hace 6 meses",
    "BILL_AMT1": "Importe facturado el último mes",
    "BILL_AMT2": "Importe facturado hace 2 meses",
    "BILL_AMT3": "Importe facturado hace 3 meses",
    "BILL_AMT4": "Importe facturado hace 4 meses",
    "BILL_AMT5": "Importe facturado hace 5 meses",
    "BILL_AMT6": "Importe facturado hace 6 meses",
    "PAY_AMT1": "Importe pagado el último mes",
    "PAY_AMT2": "Importe pagado hace 2 meses",
    "PAY_AMT3": "Importe pagado hace 3 meses",
    "PAY_AMT4": "Importe pagado hace 4 meses",
    "PAY_AMT5": "Importe pagado hace 5 meses",
    "PAY_AMT6": "Importe pagado hace 6 meses",
    "SEX": "Sexo",
    "AGE": "Edad",
    "MARRIAGE": "Estado civil",
}


def reason_text(feature: str, value: float, medians: pd.Series) -> str:
    """Traduce (variable, valor) a un reason code legible.

    La dirección ("elevado"/"reducido") se determina comparando con la mediana
    de la cartera, no asumida: un SHAP positivo puede venir tanto de facturación
    alta como de un patrón de cuenta inactiva con importes mínimos.
    """
    if feature.startswith("PAY_") and not feature.startswith("PAY_AMT"):
        v = int(value)
        if v >= 1:
            return f"{LABELS[feature]}: {v} mes(es) de retraso"
        return f"{LABELS[feature]}: patrón de pago desfavorable"
    if feature.startswith("PAY_AMT"):
        side = "reducido" if value <= medians[feature] else "con patrón desfavorable"
        return f"{LABELS[feature]} {side} ({value:,.0f} NT$)"
    if feature.startswith("BILL_AMT"):
        side = "elevado" if value >= medians[feature] else "con patrón de uso atípico"
        return f"{LABELS[feature]} {side} ({value:,.0f} NT$)"
    if feature == "LIMIT_BAL":
        side = "bajo" if value <= medians[feature] else "con uso desfavorable"
        return f"Límite de crédito {side} ({value:,.0f} NT$)"
    return f"{LABELS.get(feature, feature)}: {value:g}"


def reason_codes(x_row: pd.Series, shap_row: np.ndarray, features: list, medians: pd.Series, top_k: int = 3) -> list:
    """Top-k factores que EMPUJAN AL RIESGO (SHAP > 0), excluyendo atributos sensibles."""
    order = np.argsort(shap_row)[::-1]
    codes = []
    for idx in order:
        feat = features[idx]
        if shap_row[idx] <= 0 or feat in EXCLUDED_FROM_REASONS:
            continue
        codes.append({
            "feature": feat,
            "reason": reason_text(feat, x_row[feat], medians),
            "shap": round(float(shap_row[idx]), 4),
        })
        if len(codes) == top_k:
            break
    return codes


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
