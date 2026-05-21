from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging

from .schemas import Transaction, PredictionResponse, HealthResponse

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'app
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "API de détection de fraude bancaire en temps réel. "
        "Utilise un modèle XGBoost entraîné sur le dataset Credit Card Fraud Detection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemins des fichiers
MODEL_PATH = Path("models/model.pkl")
SCALER_PATH = Path("data/processed/scaler.pkl")
THRESHOLD = 0.5

# Variables globales pour le modèle
model = None
scaler = None


@app.on_event("startup")
async def load_model():
    """Charge le modèle et le scaler au démarrage de l'application."""
    global model, scaler
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Modèle et scaler chargés avec succès")
    else:
        logger.warning("⚠️  Modèle ou scaler non trouvé — API en mode dégradé")


@app.get("/", tags=["Root"])
def root():
    """Point d'entrée racine de l'API."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Vérifie l'état de santé de l'API et si le modèle est chargé."""
    return HealthResponse(
        status="ok",
        model_version="1.0.0",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: Transaction):
    """
    Prédit si une transaction bancaire est frauduleuse.

    - **is_fraud**: True si la transaction est considérée comme fraude
    - **fraud_probability**: Probabilité de fraude entre 0 et 1
    - **risk_level**: LOW (<0.3) | MEDIUM (0.3-0.7) | HIGH (>0.7)
    - **transaction_id**: Identifiant unique de la prédiction
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Veuillez entraîner le modèle d'abord."
        )

    try:
        # Préparer les features dans le bon ordre
        data = pd.DataFrame([transaction.dict()])
        data[["Amount", "Time"]] = scaler.transform(data[["Amount", "Time"]])

        # Prédiction
        proba = float(model.predict_proba(data)[0][1])
        is_fraud = proba >= THRESHOLD

        # Niveau de risque
        if proba < 0.3:
            risk = "LOW"
        elif proba < 0.7:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        logger.info(
            f"Prédiction: fraud={is_fraud}, proba={proba:.4f}, "
            f"risk={risk}, amount={transaction.Amount}"
        )

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(proba, 4),
            risk_level=risk,
            transaction_id=str(int(time.time() * 1000))
        )

    except Exception as e:
        logger.error(f"Erreur de prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(transactions: list[Transaction]):
    """
    Prédit pour un lot de transactions (max 100 par requête).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    if len(transactions) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 transactions par requête batch"
        )

    results = []
    data = pd.DataFrame([t.dict() for t in transactions])
    data[["Amount", "Time"]] = scaler.transform(data[["Amount", "Time"]])

    probas = model.predict_proba(data)[:, 1]
    ts_base = int(time.time() * 1000)

    for i, proba in enumerate(probas):
        proba = float(proba)
        is_fraud = proba >= THRESHOLD
        risk = "LOW" if proba < 0.3 else "MEDIUM" if proba < 0.7 else "HIGH"
        results.append(PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(proba, 4),
            risk_level=risk,
            transaction_id=str(ts_base + i)
        ))

    return {"predictions": results, "count": len(results)}
