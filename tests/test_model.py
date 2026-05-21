import pytest
import numpy as np
import pandas as pd
from pathlib import Path


def make_fake_dataframe(n_rows: int = 100, fraud_ratio: float = 0.1):
    """Génère un DataFrame factice simulant les données creditcard."""
    np.random.seed(42)
    n_fraud = int(n_rows * fraud_ratio)
    n_normal = n_rows - n_fraud

    features = {f"V{i}": np.random.randn(n_rows) for i in range(1, 29)}
    features["Time"] = np.random.uniform(0, 172792, n_rows)
    features["Amount"] = np.abs(np.random.exponential(88, n_rows))
    features["Class"] = [1] * n_fraud + [0] * n_normal

    return pd.DataFrame(features)


# ─── Tests de preprocessing ──────────────────────────────────────────────────

def test_class_distribution():
    """Vérifie que le dataset simulé a le bon ratio de fraude."""
    df = make_fake_dataframe(1000, fraud_ratio=0.1)
    fraud_ratio = df["Class"].mean()
    assert 0.08 < fraud_ratio < 0.12


def test_features_count():
    """Vérifie que le dataset a le bon nombre de features."""
    df = make_fake_dataframe()
    assert len(df.columns) == 31  # V1-V28 + Time + Amount + Class


def test_no_missing_values():
    """Vérifie qu'il n'y a pas de valeurs manquantes."""
    df = make_fake_dataframe()
    assert df.isnull().sum().sum() == 0


# ─── Tests de normalisation ──────────────────────────────────────────────────

def test_scaler_transform():
    """Vérifie que le StandardScaler normalise correctement."""
    from sklearn.preprocessing import StandardScaler
    df = make_fake_dataframe(500)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[["Amount", "Time"]])

    # Après normalisation, moyenne ≈ 0 et std ≈ 1
    assert abs(scaled[:, 0].mean()) < 0.1
    assert abs(scaled[:, 1].mean()) < 0.1
    assert abs(scaled[:, 0].std() - 1.0) < 0.1


# ─── Tests de SMOTE ──────────────────────────────────────────────────────────

def test_smote_balancing():
    """Vérifie que SMOTE équilibre correctement les classes."""
    from imblearn.over_sampling import SMOTE
    df = make_fake_dataframe(500, fraud_ratio=0.02)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Après SMOTE, les classes doivent être équilibrées
    fraud_count = y_res.sum()
    normal_count = (y_res == 0).sum()
    assert fraud_count == normal_count


# ─── Tests de métriques ──────────────────────────────────────────────────────

def test_metrics_file_format():
    """Vérifie le format du fichier de métriques s'il existe."""
    metrics_path = Path("metrics/scores.json")
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        required_keys = ["roc_auc", "f1", "precision", "recall"]
        for key in required_keys:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0


def test_model_file_exists():
    """Vérifie que le modèle existe si le pipeline a été exécuté."""
    model_path = Path("models/model.pkl")
    if model_path.exists():
        import joblib
        model = joblib.load(model_path)
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "predict")
