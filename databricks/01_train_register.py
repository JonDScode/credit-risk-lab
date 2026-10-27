# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Risk Lab — ingesta, entrenamiento, registro y explicabilidad
# MAGIC Mismo problema que `src/` (UCI default of credit card clients, 30.000 clientes), llevado a un lakehouse:
# MAGIC tablas Delta → XGBoost + MLflow → Unity Catalog model registry → SHAP con reason codes.
# MAGIC Demo en Databricks Free Edition (serverless, uso no comercial); no es un despliegue de producción.

# COMMAND ----------

# MAGIC %pip install -q "xgboost>=2.0" "shap>=0.45"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json

import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from reasons import reason_codes  # copia de src/reasons.py, sincronizada junto al notebook

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "credit_risk_lab")
P = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
CSV = f"/Volumes/{dbutils.widgets.get('catalog')}/{dbutils.widgets.get('schema')}/raw/credit_default_raw.csv"
MODEL = f"{P}.xgb_default"
SEED = 42

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("select current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/credit_risk_lab")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingesta → `credit_raw` (Delta)
# MAGIC Delta no admite espacios en nombres de columna: `default payment next month` → `default_payment_next_month`.

# COMMAND ----------

raw = pd.read_csv(CSV)
raw.columns = [c.replace(" ", "_") for c in raw.columns]
(spark.createDataFrame(raw).write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{P}.credit_raw"))
print(f"credit_raw: {len(raw)} filas, {raw.shape[1]} columnas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Limpieza → `credit_clean` (igual que `src/prepare.py`)
# MAGIC - `default payment next month` → `default`; `PAY_0` → `PAY_1`
# MAGIC - `EDUCATION` 0/5/6 → 4; `MARRIAGE` 0 → 3
# MAGIC - `ID` se conserva como clave (para `scores`) pero no es variable del modelo.

# COMMAND ----------

# Delta no garantiza el orden de las filas; la partición estratificada depende de él, así que se fija por ID.
df = spark.table(f"{P}.credit_raw").toPandas().sort_values("ID").reset_index(drop=True)
df = df.rename(columns={"default_payment_next_month": "default", "PAY_0": "PAY_1"})
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

assert df["default"].isin([0, 1]).all() and df.isna().sum().sum() == 0
assert len(df) == 30000, len(df)
print(f"tasa de impago: {df['default'].mean():.1%}")  # esperado 22,1 %
(spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{P}.credit_clean"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Entrenamiento + MLflow (igual que `src/train.py`)
# MAGIC Nota: `train.py` usa el 20 % de test como `eval_set` del early stopping, así que las métricas son algo
# MAGIC optimistas. Se replica tal cual para que cuadren con `reports/metrics.json`.

# COMMAND ----------

FEATURES = [c for c in df.columns if c not in ("ID", "default")]
X = df[FEATURES].astype("float64")
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

params = dict(n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9,
              eval_metric="auc", early_stopping_rounds=50, random_state=SEED)


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(max(tpr - fpr))


with mlflow.start_run(run_name="xgboost") as run:
    mlflow.log_params(params)
    mlflow.set_tags({"xgboost": xgboost.__version__, "shap": shap.__version__})
    print("xgboost", xgboost.__version__, "| shap", shap.__version__)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    p_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, p_test), 4),
        "pr_auc": round(average_precision_score(y_test, p_test), 4),
        "ks": round(ks_statistic(y_test, p_test), 4),
        "brier": round(brier_score_loss(y_test, p_test), 4),
    }
    mlflow.log_metrics(metrics)
    print(metrics, "\nrepo (reports/metrics.json): roc_auc 0.7799, pr_auc 0.561, ks 0.4294, brier 0.135")

    # 4. Registro en Unity Catalog. La firma describe la salida de pyfunc (clase); las probabilidades
    # se obtienen con mlflow.xgboost.load_model(...).predict_proba, como hace 02_batch_score.
    sample = X_train.head(100)
    mlflow.xgboost.log_model(
        model, artifact_path="model",
        signature=infer_signature(sample, model.predict(sample)),
        input_example=sample.head(5),
        registered_model_name=MODEL,
    )

    # 5. SHAP: motivos de la solicitud de mayor riesgo del test (reasons.py: deterministas, sin atributos protegidos)
    shap_vals = shap.TreeExplainer(model)(X_test).values
    worst = int(np.argmax(p_test))
    codes = reason_codes(X_test.iloc[worst], shap_vals[worst], FEATURES, X_train.median())
    explanation = {"p_default": round(float(p_test[worst]), 4), "reason_codes": codes}
    mlflow.log_dict(explanation, "reason_codes_example.json")
    print(json.dumps(explanation, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Versión registrada y alias `champion`
# MAGIC Demo: el alias marca la versión que usa la puntuación por lotes; no es una promoción a producción.

# COMMAND ----------

client = MlflowClient()
# ponytail: la versión recién creada es la mayor; si hubiera ejecuciones concurrentes, buscar por run_id
version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL}'"))
client.set_registered_model_alias(MODEL, "champion", version)
print(f"{MODEL} v{version} → alias champion")
