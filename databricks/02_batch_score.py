# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Risk Lab — puntuación por lotes
# MAGIC Carga el modelo con alias `champion`, puntúa `credit_clean` y escribe `scores` con la versión usada.

# COMMAND ----------

# MAGIC %pip install -q "xgboost>=2.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "credit_risk_lab")
P = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
MODEL = f"{P}.xgb_default"

mlflow.set_registry_uri("databricks-uc")
version = int(MlflowClient().get_model_version_by_alias(MODEL, "champion").version)
model = mlflow.xgboost.load_model(f"models:/{MODEL}@champion")
features = model.get_booster().feature_names  # mismas variables y orden que en el entrenamiento

# COMMAND ----------

df = spark.table(f"{P}.credit_clean").toPandas()
out = pd.DataFrame({
    "ID": df["ID"],
    "p_default": model.predict_proba(df[features].astype("float64"))[:, 1],
    "model_version": version,
})
(spark.createDataFrame(out).withColumn("scored_at", F.current_timestamp())
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{P}.scores"))

n = spark.table(f"{P}.scores").count()
assert n == len(df), n
print(f"scores: {n} filas, modelo {MODEL} v{version}, p_default medio {out['p_default'].mean():.3f}")
