"""Smoke tests de la API de scoring:  pytest tests/ -v"""
from fastapi.testclient import TestClient

from app.main import app

GOOD_CLIENT = {
    "LIMIT_BAL": 120000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_1": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 20000, "BILL_AMT2": 18000, "BILL_AMT3": 15000,
    "BILL_AMT4": 12000, "BILL_AMT5": 10000, "BILL_AMT6": 8000,
    "PAY_AMT1": 20000, "PAY_AMT2": 18000, "PAY_AMT3": 15000,
    "PAY_AMT4": 12000, "PAY_AMT5": 10000, "PAY_AMT6": 8000,
}

RISKY_CLIENT = {
    **GOOD_CLIENT,
    "LIMIT_BAL": 20000,
    "PAY_1": 3, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2, "PAY_6": 2,
    "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
}


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_score_good_client():
    with TestClient(app) as client:
        r = client.post("/score", json=GOOD_CLIENT)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["p_default"] <= 1.0
        assert body["risk_band"] in {"bajo", "medio", "alto"}


def test_score_risky_gt_good():
    with TestClient(app) as client:
        good = client.post("/score", json=GOOD_CLIENT).json()
        risky = client.post("/score", json=RISKY_CLIENT).json()
        assert risky["p_default"] > good["p_default"]
        assert len(risky["reason_codes"]) > 0
        # Los atributos sensibles nunca son motivo de denegación
        assert all(c["feature"] not in {"SEX", "AGE", "MARRIAGE", "EDUCATION"}
                   for c in risky["reason_codes"])


def test_validation_rejects_bad_input():
    with TestClient(app) as client:
        r = client.post("/score", json={**GOOD_CLIENT, "SEX": 7})
        assert r.status_code == 422
