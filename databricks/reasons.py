# Copia de ../src/reasons.py para que el bundle sincronice solo esta carpeta; mantener igual.
"""Reason codes del scoring — lógica compartida entre análisis (explain.py) y API (app/).

Deterministas por diseño: plantillas sobre SHAP values, reproducibles ante un regulador.
Los atributos protegidos por normativa (SEX, AGE, MARRIAGE) o éticamente cuestionables
como motivo de denegación (EDUCATION) nunca aparecen como reason code.
"""
import numpy as np
import pandas as pd

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
