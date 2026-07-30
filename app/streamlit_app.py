"""Demo interactiva del scoring de riesgo — Streamlit.

Ejecutar en local:   streamlit run app/streamlit_app.py
Misma lógica que la API (app/main.py): modelo XGBoost + reason codes SHAP.
"""
import sys
from pathlib import Path

import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reasons import reason_codes  # noqa: E402

st.set_page_config(page_title="Credit Risk Lab — Demo", page_icon="🏦", layout="centered")


@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model(ROOT / "models" / "xgb.json")
    medians = pd.read_json(ROOT / "models" / "medians.json", typ="series")
    explainer = shap.TreeExplainer(model)
    features = list(model.get_booster().feature_names)
    return model, medians, explainer, features


model, medians, explainer, FEATURES = load_artifacts()

st.title("🏦 Credit Risk Lab")
st.caption(
    "Scoring de impago **explicable**: XGBoost + SHAP reason codes. "
    "Modelo entrenado sobre 30.000 clientes reales (UCI). "
    "[Código en GitHub](https://github.com/JonDScode/credit-risk-lab)"
)

PAY_OPTIONS = {
    -1: "Al día",
    0: "Uso revolving",
    1: "1 mes de retraso",
    2: "2 meses de retraso",
    3: "3 meses de retraso",
    4: "4+ meses de retraso",
}

st.subheader("Perfil del solicitante")
col1, col2 = st.columns(2)
with col1:
    limit_bal = st.number_input("Límite de crédito (NT$)", 10000, 1000000, 120000, step=10000)
    pay_1 = st.selectbox("Pago del último mes", PAY_OPTIONS.keys(), index=0,
                         format_func=PAY_OPTIONS.get)
    pay_2 = st.selectbox("Pago hace 2 meses", PAY_OPTIONS.keys(), index=0,
                         format_func=PAY_OPTIONS.get)
    pay_3 = st.selectbox("Pago hace 3 meses", PAY_OPTIONS.keys(), index=0,
                         format_func=PAY_OPTIONS.get)
with col2:
    bill_amt1 = st.number_input("Facturado el último mes (NT$)", 0, 1000000, 20000, step=5000)
    pay_amt1 = st.number_input("Pagado el último mes (NT$)", 0, 500000, 20000, step=5000)
    pay_amt2 = st.number_input("Pagado hace 2 meses (NT$)", 0, 500000, 18000, step=5000)
    pay_amt3 = st.number_input("Pagado hace 3 meses (NT$)", 0, 500000, 15000, step=5000)

with st.expander("Resto de variables (valores medianos de la cartera por defecto)"):
    st.caption(
        "Los atributos demográficos alimentan el modelo pero **nunca** aparecen como "
        "motivo de denegación (contrato de fairness, verificado por test)."
    )
    age = st.slider("Edad", 18, 80, 35)
    sex = st.selectbox("Sexo (código dataset)", [1, 2], index=1)
    education = st.selectbox("Educación (1=posgrado, 2=universidad, 3=secundaria, 4=otros)", [1, 2, 3, 4], index=1)
    marriage = st.selectbox("Estado civil (1=casado/a, 2=soltero/a, 3=otros)", [1, 2, 3], index=1)

client = {f: float(medians[f]) for f in FEATURES}
client.update({
    "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage,
    "AGE": age, "PAY_1": pay_1, "PAY_2": pay_2, "PAY_3": pay_3,
    "BILL_AMT1": bill_amt1, "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
})

row = pd.DataFrame([client])[FEATURES]
p = float(model.predict_proba(row)[0, 1])
band = "bajo" if p < 0.15 else ("medio" if p < 0.40 else "alto")
color = {"bajo": "🟢", "medio": "🟡", "alto": "🔴"}[band]

st.divider()
m1, m2 = st.columns(2)
m1.metric("Probabilidad de impago", f"{p:.1%}")
m2.metric("Banda de riesgo", f"{color} {band}")
st.progress(min(p / 0.6, 1.0))

shap_row = explainer(row).values[0]
codes = reason_codes(row.iloc[0], shap_row, FEATURES, medians)

st.subheader("¿Por qué? — reason codes")
if codes:
    for c in codes:
        st.markdown(f"- {c['reason']}  &nbsp; `SHAP +{c['shap']:.2f}`")
else:
    st.markdown("Sin factores de riesgo relevantes: el perfil empuja la predicción a la baja.")

st.caption(
    "Reason codes deterministas (plantillas sobre SHAP values, reproducibles ante un "
    "regulador). La dirección de cada razón se calcula contra la mediana de la cartera. "
    "Métricas del modelo: AUC 0.78 · KS 0.43 en test."
)
