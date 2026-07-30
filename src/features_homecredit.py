"""Fase 5 — Feature engineering multi-tabla sobre Home Credit Default Risk.

De 7 tablas relacionales a una tabla de modelado por solicitud (SK_ID_CURR).
Features CURADAS y legibles — no una explosión automática de agregados: cada
una tiene una lectura de negocio que se puede defender en un comité de riesgo.

Entrada:  data/raw/homecredit/*.csv   (kaggle competitions download)
Salida:   data/processed/homecredit_features.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "homecredit"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "homecredit_features.csv"

# Anomalía documentada del dataset: DAYS_EMPLOYED = 365243 significa "sin empleo registrado"
EMPLOYED_ANOMALY = 365243


def application_features() -> pd.DataFrame:
    """Tabla principal: una fila por solicitud, con ratios de capacidad de pago."""
    cols = [
        "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "CNT_CHILDREN", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_CONTRACT_TYPE",
        "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "OCCUPATION_TYPE",
    ]
    df = pd.read_csv(RAW / "application_train.csv", usecols=cols)

    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(EMPLOYED_ANOMALY, np.nan)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    # Ratios de capacidad de pago — el corazón de cualquier scorecard
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TO_GOODS"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["CREDIT_TERM_MONTHS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

    df["FLAG_OWN_CAR"] = (df["FLAG_OWN_CAR"] == "Y").astype(int)
    df["FLAG_OWN_REALTY"] = (df["FLAG_OWN_REALTY"] == "Y").astype(int)

    # Categóricas de baja cardinalidad -> one-hot
    df = pd.get_dummies(
        df,
        columns=["NAME_CONTRACT_TYPE", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "OCCUPATION_TYPE"],
        dummy_na=True,
        dtype=int,
    )
    return df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])


def bureau_features() -> pd.DataFrame:
    """Historial en OTRAS entidades (buró de crédito): endeudamiento y morosidad externa."""
    bureau = pd.read_csv(RAW / "bureau.csv", usecols=[
        "SK_ID_CURR", "CREDIT_ACTIVE", "DAYS_CREDIT", "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "CREDIT_DAY_OVERDUE",
    ])
    g = bureau.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "BUREAU_N_CREDITS": g.size(),
        "BUREAU_N_ACTIVE": bureau.assign(a=(bureau["CREDIT_ACTIVE"] == "Active").astype(int)).groupby("SK_ID_CURR")["a"].sum(),
        "BUREAU_TOTAL_DEBT": g["AMT_CREDIT_SUM_DEBT"].sum(),
        "BUREAU_TOTAL_CREDIT": g["AMT_CREDIT_SUM"].sum(),
        "BUREAU_OVERDUE_AMT": g["AMT_CREDIT_SUM_OVERDUE"].sum(),
        "BUREAU_MAX_DAYS_OVERDUE": g["CREDIT_DAY_OVERDUE"].max(),
        "BUREAU_DAYS_SINCE_LAST": g["DAYS_CREDIT"].max(),
    })
    out["BUREAU_DEBT_RATIO"] = out["BUREAU_TOTAL_DEBT"] / out["BUREAU_TOTAL_CREDIT"].replace(0, np.nan)
    return out.reset_index()


def previous_app_features() -> pd.DataFrame:
    """Solicitudes PREVIAS en Home Credit: cuántas y cuántas aprobadas."""
    prev = pd.read_csv(RAW / "previous_application.csv", usecols=[
        "SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT", "CNT_PAYMENT",
    ])
    g = prev.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "PREV_N_APPS": g.size(),
        "PREV_APPROVAL_RATE": prev.assign(a=(prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)).groupby("SK_ID_CURR")["a"].mean(),
        "PREV_MEAN_PAYMENTS": g["CNT_PAYMENT"].mean(),
        "PREV_CREDIT_TO_APPLICATION": (prev["AMT_CREDIT"] / prev["AMT_APPLICATION"].replace(0, np.nan)).groupby(prev["SK_ID_CURR"]).mean(),
    })
    return out.reset_index()


def installments_features() -> pd.DataFrame:
    """Comportamiento de pago de cuotas: retrasos e infrapagos reales."""
    inst = pd.read_csv(RAW / "installments_payments.csv", usecols=[
        "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
    ], dtype={"AMT_INSTALMENT": np.float32, "AMT_PAYMENT": np.float32})
    inst["DAYS_LATE"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["UNDERPAID"] = (inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"] * 0.99).astype(int)
    g = inst.groupby("SK_ID_CURR")
    return pd.DataFrame({
        "INST_MEAN_DAYS_LATE": g["DAYS_LATE"].mean(),
        "INST_MAX_DAYS_LATE": g["DAYS_LATE"].max(),
        "INST_UNDERPAY_RATE": g["UNDERPAID"].mean(),
        "INST_N_PAYMENTS": g.size(),
    }).reset_index()


def balances_features() -> pd.DataFrame:
    """Saldos mensuales (POS y tarjeta): días en mora y utilización de tarjeta."""
    pos = pd.read_csv(RAW / "POS_CASH_balance.csv", usecols=["SK_ID_CURR", "SK_DPD"])
    pos_g = pos.groupby("SK_ID_CURR")["SK_DPD"].agg(POS_MEAN_DPD="mean", POS_MAX_DPD="max")

    cc = pd.read_csv(RAW / "credit_card_balance.csv", usecols=[
        "SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD",
    ], dtype={"AMT_BALANCE": np.float32, "AMT_CREDIT_LIMIT_ACTUAL": np.float32})
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    cc_g = cc.groupby("SK_ID_CURR").agg(
        CC_MEAN_UTILIZATION=("UTILIZATION", "mean"),
        CC_MAX_DPD=("SK_DPD", "max"),
    )
    return pos_g.join(cc_g, how="outer").reset_index()


def main() -> None:
    print("application ...")
    df = application_features()
    for name, fn in [
        ("bureau", bureau_features),
        ("previous_application", previous_app_features),
        ("installments", installments_features),
        ("balances", balances_features),
    ]:
        print(f"{name} ...")
        df = df.merge(fn(), on="SK_ID_CURR", how="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"OK: {OUT} — {df.shape[0]} solicitudes, {df.shape[1]} columnas, "
          f"tasa de impago {df['TARGET'].mean():.1%}")


if __name__ == "__main__":
    main()
