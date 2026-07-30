"""Demo interactiva del scoring de riesgo — Streamlit.

Ejecutar en local:   streamlit run app/streamlit_app.py
Misma lógica que la API (app/main.py): modelo XGBoost + reason codes SHAP.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reasons import reason_codes  # noqa: E402

st.set_page_config(page_title="Credit Risk Lab", page_icon=":material/account_balance:",
                   layout="wide")


@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model(ROOT / "models" / "xgb.json")
    medians = pd.read_json(ROOT / "models" / "medians.json", typ="series")
    explainer = shap.TreeExplainer(model)
    features = list(model.get_booster().feature_names)
    dist = json.loads((ROOT / "reports" / "score_distribution.json").read_text())
    return model, medians, explainer, features, dist


model, medians, explainer, FEATURES, DIST = load_artifacts()

st.title(":material/account_balance: Credit Risk Lab")
st.markdown(
    "**Scoring de impago explicable**: XGBoost + SHAP reason codes, entrenado sobre "
    "30.000 clientes reales. &nbsp; [:material/code: Código](https://github.com/JonDScode/credit-risk-lab) · "
    "[:material/language: jonnathanospina.com](https://www.jonnathanospina.com)"
)

with st.container(border=True):
    c1, c2 = st.columns(2)
    c1.markdown(
        ":material/support_agent: **Si eres analista o asesor de riesgo** — esto simula "
        "la herramienta de decisión que usarías en tu entidad: puntúa una solicitud y "
        "justifica cada decisión con motivos trazables, como exige la normativa (EBA / AI Act)."
    )
    c2.markdown(
        ":material/person: **Si eres una persona de a pie** — así decide un banco si te "
        "concede un crédito: mira tu historial de pagos reciente, no tu sexo ni tu edad. "
        "Prueba a cambiar los pagos y verás por qué."
    )

tab_score, tab_model = st.tabs([":material/person_search: Evaluar solicitante",
                                ":material/monitoring: Sobre el modelo"])

PAY_OPTIONS = {
    -1: "Al día",
    0: "Uso revolving",
    1: "1 mes de retraso",
    2: "2 meses de retraso",
    3: "3 meses de retraso",
    4: "4+ meses de retraso",
}

with tab_score:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Perfil del solicitante")
        limit_bal = st.number_input("Límite de crédito (NT$)", 10000, 1000000, 120000, step=10000)
        f1, f2 = st.columns(2)
        with f1:
            pay_1 = st.selectbox("Pago del último mes", PAY_OPTIONS.keys(), index=0,
                                 format_func=PAY_OPTIONS.get)
            pay_2 = st.selectbox("Pago hace 2 meses", PAY_OPTIONS.keys(), index=0,
                                 format_func=PAY_OPTIONS.get)
            pay_3 = st.selectbox("Pago hace 3 meses", PAY_OPTIONS.keys(), index=0,
                                 format_func=PAY_OPTIONS.get)
            bill_amt1 = st.number_input("Facturado el último mes (NT$)", 0, 1000000, 20000, step=5000)
        with f2:
            pay_amt1 = st.number_input("Pagado el último mes (NT$)", 0, 500000, 20000, step=5000)
            pay_amt2 = st.number_input("Pagado hace 2 meses (NT$)", 0, 500000, 18000, step=5000)
            pay_amt3 = st.number_input("Pagado hace 3 meses (NT$)", 0, 500000, 15000, step=5000)

        with st.expander("Resto de variables (medianas de la cartera por defecto)"):
            st.caption(
                "Los atributos demográficos alimentan el modelo pero **nunca** aparecen "
                "como motivo de denegación (contrato de fairness, verificado por test)."
            )
            age = st.slider("Edad", 18, 80, 35)
            sex = st.selectbox("Sexo (código dataset)", [1, 2], index=1)
            education = st.selectbox("Educación (1=posgrado, 2=universidad, 3=secundaria, 4=otros)",
                                     [1, 2, 3, 4], index=1)
            marriage = st.selectbox("Estado civil (1=casado/a, 2=soltero/a, 3=otros)",
                                    [1, 2, 3], index=1)

    client = {f: float(medians[f]) for f in FEATURES}
    client.update({
        "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage,
        "AGE": age, "PAY_1": pay_1, "PAY_2": pay_2, "PAY_3": pay_3,
        "BILL_AMT1": bill_amt1, "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
    })
    row = pd.DataFrame([client])[FEATURES]
    p = float(model.predict_proba(row)[0, 1])
    band = "bajo" if p < 0.15 else ("medio" if p < 0.40 else "alto")

    with col_result:
        st.subheader("Resultado")
        m1, m2 = st.columns(2)
        m1.metric("Probabilidad de impago", f"{p:.1%}")
        with m2:
            st.markdown("Banda de riesgo")
            if band == "bajo":
                st.badge("Riesgo bajo", icon=":material/check_circle:", color="green")
            elif band == "medio":
                st.badge("Riesgo medio", icon=":material/warning:", color="orange")
            else:
                st.badge("Riesgo alto", icon=":material/error:", color="red")

        # Dónde cae este solicitante dentro de la cartera de test
        fig, ax = plt.subplots(figsize=(7, 2.6))
        edges = DIST["edges"]
        ax.bar(edges[:-1], DIST["counts"], width=edges[1] - edges[0],
               align="edge", color="#b8c4d0", edgecolor="none")
        ax.axvline(p, color="#c0392b", lw=2)
        ax.annotate(f"  este solicitante ({p:.0%})", xy=(p, max(DIST["counts"]) * 0.85),
                    color="#c0392b", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Probabilidad de impago — distribución en 6.000 solicitudes de test",
                      fontsize=8.5)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("##### :material/help_center: ¿Por qué? — reason codes")
        shap_row = explainer(row).values[0]
        codes = reason_codes(row.iloc[0], shap_row, FEATURES, medians)
        if codes:
            for c in codes:
                st.markdown(f"- {c['reason']} &nbsp; `SHAP +{c['shap']:.2f}`")
        else:
            st.markdown("Sin factores de riesgo relevantes: el perfil empuja la predicción a la baja.")

with tab_model:
    st.subheader("Métricas (test 20%, sin fuga de datos)")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("ROC AUC", "0.780")
    t2.metric("KS statistic", "0.429", help="KS > 0.4 = scorecard fuerte según el estándar del sector")
    t3.metric("PR AUC", "0.561")
    t4.metric("Brier", "0.135", help="Calibración: menor es mejor")

    g1, g2 = st.columns(2)
    with g1:
        st.image(str(ROOT / "reports" / "figures" / "roc.png"),
                 caption="ROC — XGBoost vs regresión logística")
    with g2:
        st.image(str(ROOT / "reports" / "figures" / "shap_summary.png"),
                 caption="SHAP — qué mueve el riesgo en la cartera")

    st.markdown(
        """
        ##### :material/balance: Decisiones de diseño
        - **Reason codes deterministas** — plantillas sobre SHAP values, no un LLM:
          una denegación debe ser reproducible ante un regulador.
        - **Fairness testeada** — sexo, edad, estado civil y educación nunca aparecen
          como motivo de denegación; hay un test que falla si eso cambia.
        - **Dirección calculada** — "elevado"/"reducido" se contrasta con la mediana
          de la cartera, no se asume.
        - **DL comparado y descartado con datos** — un MLP en PyTorch quedó por debajo
          de XGBoost (AUC 0.771 vs 0.780) y peor calibrado, consistente con la literatura.
        - En la fase Home Credit (307.000 préstamos, 7 tablas), el mismo protocolo
          alcanza **AUC 0.783** con features interpretables.
        """
    )
