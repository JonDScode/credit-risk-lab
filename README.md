# 🏦 Credit Risk Lab

Laboratorio de **scoring de riesgo crediticio** de principio a fin: desde los datos crudos
hasta un modelo explicable servido por API — el problema central de la banca, resuelto
con las prácticas que exige el sector (explicabilidad, trazabilidad, despliegue).

## ¿Por qué este proyecto?

El scoring de crédito es el caso de uso de ML más regulado y maduro de la industria
financiera. Un modelo que predice bien no basta: la normativa (EBA, AI Act) exige poder
**explicar cada decisión**. Este proyecto ataca exactamente eso:

- **Modelo**: predicción de impago (probabilidad de default) con Gradient Boosting.
- **Explicabilidad**: SHAP values globales (qué mueve el riesgo en la cartera) y locales
  (por qué se deniega *esta* solicitud), traducidos a *reason codes*.
- **Producción**: API de scoring con FastAPI + Docker, con el patrón de un servicio real.

## Datos

| Fase | Dataset | Tamaño |
|------|---------|--------|
| Actual | [UCI — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Taiwán, 2005) | 30.000 clientes, 23 variables |
| Próxima | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle) | 307.000 préstamos, multi-tabla |

Los datos **no se versionan**: `python src/download_data.py` los descarga y deja en `data/raw/`.

## Estructura

```
credit-risk-lab/
├── src/
│   ├── download_data.py    # Descarga y validación del dataset
│   ├── prepare.py          # Limpieza y features → data/processed/
│   └── train.py            # Entrenamiento + métricas + SHAP
├── notebooks/              # EDA y análisis
├── app/                    # (fase 3) API FastAPI de scoring
├── reports/figures/        # Gráficas generadas
└── data/                   # raw/ y processed/ (gitignored)
```

## Uso

```bash
pip install -r requirements.txt
python src/download_data.py
python src/prepare.py
python src/train.py
```

## Roadmap

- [x] Fase 1 — Datos + EDA + baseline XGBoost con métricas honestas (AUC, KS, calibración)
- [ ] Fase 2 — SHAP: explicabilidad global y local, reason codes en lenguaje natural
- [ ] Fase 3 — API de scoring (FastAPI + Docker) con endpoint `/score` explicado
- [ ] Fase 4 — Variante Deep Learning tabular (PyTorch) y comparativa rigurosa
- [ ] Fase 5 — Upgrade a Home Credit (multi-tabla, feature engineering serio)
- [ ] Demo Streamlit + artículo en [el blog](https://www.jonnathanospina.com/pages/blog.html)

---

**Jonnathan Ospina** — [jonnathanospina.com](https://www.jonnathanospina.com) · [LinkedIn](https://www.linkedin.com/in/jonnathanospinam)
