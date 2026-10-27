# Credit Risk Lab en Databricks — alcance y estado (27-oct-2026)

Objetivo: demostrar que el mismo problema de scoring se puede llevar a un lakehouse (tablas Delta, MLflow, un job),
sin rehacer el modelo. No sustituye la API de AWS (`Dockerfile`, FastAPI): la complementa. Entrega: martes 27-oct,
para la entrevista.

**Estado:** pasos 1–7 implementados y ejecutados en el workspace (jobs `train` y `score` en verde). Faltan las
capturas del §4.

## 1. Qué se hace y qué no

**Sí:**
- Cargar el dataset de UCI (30.000 clientes, `data/raw/credit_default_raw.csv`) en una tabla Delta gestionada.
- Entrenar XGBoost con las mismas variables e hiperparámetros que `src/train.py` y registrar el experimento en MLflow.
- Registrar el modelo con sus métricas (ROC AUC, PR AUC, KS, Brier) y una versión.
- Calcular SHAP para una solicitud y devolver los motivos en texto, con la lógica de `src/reasons.py`.
- Puntuar la tabla completa por lotes y guardar el resultado en otra tabla Delta.
- Un job que ejecute la puntuación.

**No:**
- Home Credit (307.511 préstamos, siete tablas): demasiado para la capa gratuita y no aporta a la demo.
- Endpoint de serving: no está verificado en la capa gratuita. Si aparece, es un extra.
- Red neuronal (`train_torch.py`): el propio repo justifica que XGBoost gana en tabular de este tamaño.
- Datos reales de clientes: solo el dataset público de UCI.

## 2. Entorno

- **Databricks Free Edition** ([limitaciones](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)):
  solo cómputo serverless, sin GPU, hasta 5 tareas simultáneas en jobs, sin SLA, uso no comercial.
- **Catálogo y esquema:** `workspace.credit_risk_lab` (Free Edition ya trae el catálogo `workspace`; no se crea uno
  propio). Objetos: volumen `raw`, tablas `credit_raw`, `credit_clean` y `scores`, modelo `xgb_default`.
- **Datos:** el CSV no está en git (`data/raw/` y `data/processed/` están ignorados). Se regenera con
  `src/download_data.py` y se sube al volumen `raw` con `databricks fs cp`.
- **Librerías:** `xgboost` y `shap` se instalan con `%pip` al inicio de cada notebook. Versiones de la última
  ejecución: xgboost 3.4.1, shap 0.54.1 (local: xgboost 3.3.0). Se guardan como tags de la ejecución en MLflow.
- **Despliegue:** `databricks/databricks.yml` (Declarative Automation Bundles) define los dos jobs, sin clúster
  (serverless) y con `performance_target: STANDARD`. Perfil del CLI: `credit-risk`.

```bash
cd databricks
databricks bundle validate --strict -t dev --profile credit-risk
databricks bundle deploy -t dev --profile credit-risk
databricks bundle run train_job -t dev --profile credit-risk
databricks bundle run score_job -t dev --profile credit-risk
```

## 3. Pasos

Dos notebooks en formato fuente `.py` (`databricks/`): `01_train_register.py` (pasos 1–5) y `02_batch_score.py` (paso 6).

1. **Ingesta.** Leer el CSV del volumen, cambiar espacios por `_` en los nombres de columna (Delta no admite espacios)
   y escribir `credit_raw`. Referencia: `src/prepare.py`.
2. **Limpieza.** Derivar `credit_clean` igual que `src/prepare.py`:
   - `default payment next month` → `default` y `PAY_0` → `PAY_1`.
   - `EDUCATION` 0, 5 y 6 → 4; `MARRIAGE` 0 → 3.
   - `ID` se **conserva** como clave para `scores` (25 columnas), pero no es variable del modelo (23 variables).
   - Comprobar 30.000 filas y una tasa de impago de 22,1 %.
   - **Ordenar por `ID`.** Delta no garantiza el orden de las filas y la partición estratificada depende de él; sin
     orden fijo el test cambia y las métricas no son comparables con el repo (la primera ejecución dio AUC 0,7959).
3. **Entrenamiento.** Partición 80/20 estratificada con `random_state=42`. XGBoost con `n_estimators=600`,
   `learning_rate=0.05`, `max_depth=4`, `subsample=0.9`, `colsample_bytree=0.9`, `eval_metric="auc"`,
   `early_stopping_rounds=50`. Registrar en MLflow parámetros y las cuatro métricas. Referencia: `src/train.py`.
   *Nota:* `train.py` usa el test como `eval_set` del early stopping, así que las métricas son algo optimistas. Se
   replica tal cual para que cuadren con `reports/metrics.json`; si se separara un conjunto de validación, el AUC
   bajaría un poco.
4. **Registro.** Modelo `workspace.credit_risk_lab.xgb_default` en Unity Catalog, con el alias `champion` en la
   versión usada. Es solo una demo; no se llama «producción».
5. **Explicabilidad.** SHAP (`TreeExplainer`) sobre el test; para la solicitud de mayor riesgo, los tres motivos
   principales en texto con `reasons.py` (sin atributos protegidos). Se guarda como artefacto
   `reason_codes_example.json` en la ejecución.
6. **Puntuación por lotes.** Cargar `models:/…xgb_default@champion`, puntuar `credit_clean` y escribir `scores`
   (`ID`, `p_default`, `model_version`, `scored_at`).
7. **Job.** `score_job`: una tarea que ejecuta `02_batch_score.py`. `train_job` ejecuta `01_train_register.py`.

`databricks/reasons.py` es una copia de `src/reasons.py` y debe mantenerse igual. El bundle no puede sincronizar
`../src` mientras la carpeta `.pytest_cache` de la raíz tenga permisos rotos (acceso denegado).

## 4. Capturas y artefactos para la entrevista

Guardar en `reports/databricks/` (no en el notebook fuente):

| # | Qué capturar | Dónde |
|---|---|---|
| 1 | `credit_clean` con su número de filas (30.000) | Catalog → `workspace` → `credit_risk_lab` → `credit_clean` → pestaña *Details* / *Sample data* |
| 2 | Experimento con las métricas | Experiments → `credit_risk_lab` → última ejecución (ROC AUC 0,7802, KS 0,4342) |
| 3 | El modelo con su versión y el alias `champion` | Catalog → `credit_risk_lab` → Models → `xgb_default` |
| 4 | Los motivos SHAP en texto | Salida de la celda de explicabilidad de `01_train_register`, o el artefacto `reason_codes_example.json` |
| 5 | El job completado con su duración | Jobs & Pipelines → `credit-risk-batch-score` → última ejecución |
| 6 | La tabla `scores` con `model_version` | Catalog → `scores` → *Sample data* |
| 7 | Notebook exportado **con salidas** | Abrir la ejecución del job → *Export* (`.html`), o *File → Export* |

Sin datos de clientes en las capturas: los datos son públicos, pero se muestran agregados o filas sin nombres.

## 5. Criterios de hecho

- Los notebooks corren de principio a fin en serverless sin errores, con la semilla fijada. ✅
- Las métricas coinciden con `reports/metrics.json` con tolerancia pequeña. ✅

  | Métrica | Databricks | Repo |
  |---|---|---|
  | ROC AUC | 0,7802 | 0,7799 |
  | PR AUC | 0,5566 | 0,5610 |
  | KS | 0,4342 | 0,4294 |
  | Brier | 0,1350 | 0,1350 |
- El job de puntuación escribe `scores` con el número de versión del modelo en cada fila (30.000 filas, versión 2). ✅
- Capturas guardadas en `reports/databricks/`. ⬜

## 6. Qué decir en la entrevista

- «Mismo problema en dos entornos: un contenedor con FastAPI en AWS para servir, y un lakehouse en Databricks para
  entrenar, versionar y puntuar por lotes.»
- «La versión gratuita tiene límites; lo declaro y no lo vendo como producción.»
- «Elegí XGBoost porque el repo lo mide: el MLP queda por detrás en discriminación (AUC 0,771 frente a 0,780) y en
  calibración (Brier 0,182 frente a 0,135).»
- «Mi primera ejecución dio un AUC mejor, 0,796, y desconfié: el orden de filas de Delta cambiaba la partición. Fijé
  el orden por ID y volvió a 0,780.»
- «El early stopping usa el test, así que las métricas son algo optimistas; lo replico para poder compararlas y lo
  declaro.»

## 7. Riesgos

| Riesgo | Qué hacer |
|---|---|
| Cuota agotada a mitad de la demo | Ejecutar los jobs antes de la entrevista y guardar las salidas (capturas y `.html`). |
| Diferencias de métricas con el repo | Misma semilla, mismas variables y orden por `ID`; si difiere, documentar por qué en vez de ajustar. |
| Cada ejecución tarda ~4–5 min (arranque serverless e instalación de librerías) | No ejecutar en directo; mostrar la ejecución ya hecha. |
| Confundir demo con producción | Lo dice el propio documento, y se dice en la entrevista. |
| `databricks/reasons.py` se desincroniza de `src/reasons.py` | Copiar de nuevo al cambiar `src/reasons.py`. |
