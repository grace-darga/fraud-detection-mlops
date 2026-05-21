from pydantic import BaseModel, Field
from typing import Optional


class Transaction(BaseModel):
    """Représente une transaction bancaire pour la prédiction."""

    Time: float = Field(..., description="Secondes depuis la première transaction du dataset")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., description="Montant de la transaction en euros", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0,
                "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
                "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
                "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
                "V17": 0.21, "V18": 0.02, "V19": 0.40, "V20": 0.25,
                "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
                "Amount": 149.62
            }
        }


class PredictionResponse(BaseModel):
    """Résultat de la prédiction de fraude."""
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH")
    transaction_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Statut de santé de l'API."""
    status: str
    model_version: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée."""
    detail: str
