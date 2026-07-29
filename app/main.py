"""Fase 3 — API de scoring de riesgo crediticio.

POST /score : probabilidad de impago + banda de riesgo + reason codes (SHAP).
GET  /health: estado del servicio y del modelo.

Arranque:  uvicorn app.main:app --reload      (desde la raíz del repo)
Docker:    docker build -t credit-risk-api . && docker run -p 8000:8000 credit-risk-api
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from reasons import reason_codes  # noqa: E402

MODEL_PATH = ROOT / "models" / "xgb.json"
MEDIANS_PATH = ROOT / "models" / "medians.json"

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    state["model"] = model
    state["features"] = list(model.get_booster().feature_names)
    state["medians"] = pd.read_json(MEDIANS_PATH, typ="series")
    state["explainer"] = shap.TreeExplainer(model)
    yield
    state.clear()


app = FastAPI(
    title="Credit Risk Lab — Scoring API",
    description="Scoring de impago explicable: XGBoost + SHAP reason codes.",
    version="0.1.0",
    lifespan=lifespan,
)


class ClientData(BaseModel):
    """Solicitud de scoring. Códigos según el dataset UCI credit default."""

    LIMIT_BAL: float = Field(gt=0, description="Límite de crédito concedido (NT$)")
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=1, le=4)
    MARRIAGE: int = Field(ge=1, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY_1: int = Field(ge=-2, le=8, description="Meses de retraso último mes (-1 = al día)")
    PAY_2: int = Field(ge=-2, le=8)
    PAY_3: int = Field(ge=-2, le=8)
    PAY_4: int = Field(ge=-2, le=8)
    PAY_5: int = Field(ge=-2, le=8)
    PAY_6: int = Field(ge=-2, le=8)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class ReasonCode(BaseModel):
    feature: str
    reason: str
    shap: float


class ScoreResponse(BaseModel):
    p_default: float
    risk_band: str
    reason_codes: list[ReasonCode]


def risk_band(p: float) -> str:
    if p < 0.15:
        return "bajo"
    if p < 0.40:
        return "medio"
    return "alto"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH.name, "features": len(state["features"])}


@app.post("/score", response_model=ScoreResponse)
def score(client: ClientData) -> ScoreResponse:
    row = pd.DataFrame([client.model_dump()])[state["features"]]
    p = float(state["model"].predict_proba(row)[0, 1])
    shap_row = state["explainer"](row).values[0]
    codes = reason_codes(row.iloc[0], shap_row, state["features"], state["medians"])
    return ScoreResponse(p_default=round(p, 4), risk_band=risk_band(p), reason_codes=codes)
