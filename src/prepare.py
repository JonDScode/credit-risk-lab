"""Limpieza y preparación: data/raw/ -> data/processed/credit_default.csv.

Decisiones documentadas:
- Target renombrado a `default` (1 = impago el mes siguiente, ~22% de la muestra).
- EDUCATION: los códigos 0, 5 y 6 no están en el diccionario oficial -> se agrupan en 4 ("others").
- MARRIAGE: el código 0 no está documentado -> se agrupa en 3 ("others").
- PAY_0 se renombra PAY_1 para que la serie temporal sea consistente (PAY_1..PAY_6).
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw" / "credit_default_raw.csv"
OUT = DATA / "processed" / "credit_default.csv"


def main() -> None:
    df = pd.read_csv(RAW)

    df = df.rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"})
    df = df.drop(columns=["ID"])

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    assert df["default"].isin([0, 1]).all()
    assert df.isna().sum().sum() == 0, "Hay nulos inesperados"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    rate = df["default"].mean()
    print(f"OK: {OUT} ({df.shape[0]} filas, {df.shape[1]} cols, tasa de impago {rate:.1%})")


if __name__ == "__main__":
    main()
