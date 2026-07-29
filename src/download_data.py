"""Descarga el dataset UCI 'Default of Credit Card Clients' a data/raw/.

Fuente: UCI Machine Learning Repository (id 350).
30.000 clientes de tarjetas de crédito (Taiwán, 2005), objetivo: impago del mes siguiente.
"""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
OUT = RAW_DIR / "credit_default_raw.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"Ya existe {OUT} — nada que hacer.")
        return

    print(f"Descargando {URL} ...")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xls_name = next(n for n in zf.namelist() if n.lower().endswith(".xls"))
        with zf.open(xls_name) as fh:
            # La primera fila del xls es un título; la cabecera real es la segunda.
            df = pd.read_excel(fh, header=1)

    assert df.shape[0] == 30000, f"Se esperaban 30000 filas, llegaron {df.shape[0]}"
    df.to_csv(OUT, index=False)
    print(f"OK: {OUT} ({df.shape[0]} filas, {df.shape[1]} columnas)")


if __name__ == "__main__":
    main()
