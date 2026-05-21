import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from src.api.main import app

client = TestClient(app)


def make_transaction(**overrides):
    """Crée une transaction de test avec des valeurs par défaut."""
    base = {f"V{i}": 0.0 for i in range(1, 29)}
    base.update({"Time": 100.0, "Amount": 50.0})
    base.update(overrides)
    return base


# ─── Tests de santé ─────────────────────────────────────────────────────────

def test_root():
    """Vérifie que l'endpoint racine répond correctement."""
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health_endpoint():
    """Vérifie que /health retourne le bon format."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body
    assert "model_loaded" in body


# ─── Tests de prédiction ─────────────────────────────────────────────────────

def test_predict_no_model():
    """Sans modèle chargé, l'API doit retourner 503."""
    import src.api.main as api_module
    original_model = api_module.model
    api_module.model = None
    try:
        r = client.post("/predict", json=make_transaction())
        assert r.status_code == 503
    finally:
        api_module.model = original_model


def test_predict_missing_fields():
    """Une requête incomplète doit retourner une erreur de validation."""
    r = client.post("/predict", json={"Time": 100.0})
    assert r.status_code == 422


def test_predict_invalid_amount():
    """Un montant négatif doit être rejeté."""
    r = client.post("/predict", json=make_transaction(Amount=-10.0))
    assert r.status_code == 422


def test_predict_extra_fields_ignored():
    """Les champs supplémentaires doivent être ignorés sans erreur."""
    payload = make_transaction()
    payload["unknown_field"] = "ignored"
    # FastAPI ignore les champs supplémentaires par défaut
    # Ce test vérifie juste que la structure est correcte (sans modèle, 503)
    r = client.post("/predict", json=payload)
    assert r.status_code in [200, 503]  # 503 si modèle non chargé


# ─── Tests de schéma ─────────────────────────────────────────────────────────

def test_transaction_schema():
    """Vérifie que le schéma Transaction valide correctement."""
    from src.api.schemas import Transaction
    t = Transaction(**make_transaction())
    assert t.Amount == 50.0
    assert t.Time == 100.0
    assert t.V1 == 0.0


def test_prediction_response_schema():
    """Vérifie le schéma PredictionResponse."""
    from src.api.schemas import PredictionResponse
    r = PredictionResponse(
        is_fraud=False,
        fraud_probability=0.05,
        risk_level="LOW",
        transaction_id="12345"
    )
    assert r.is_fraud is False
    assert r.risk_level == "LOW"


# ─── Tests de batch ──────────────────────────────────────────────────────────

def test_batch_too_many_transactions():
    """Un batch de plus de 100 transactions doit être rejeté."""
    import src.api.main as api_module
    if api_module.model is not None:
        transactions = [make_transaction() for _ in range(101)]
        r = client.post("/predict/batch", json=transactions)
        assert r.status_code == 400
