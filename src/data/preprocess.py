import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import yaml


def preprocess(params_path: str = "params.yaml"):
    """Préprocesse les données brutes et applique SMOTE pour équilibrer les classes."""
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Charger les données
    df = pd.read_csv("data/raw/creditcard.csv")
    print(f"📊 Dataset chargé: {df.shape} — fraudes: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")

    # Séparer features / target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Normaliser Amount et Time (les features PCA V1-V28 sont déjà normalisées)
    scaler = StandardScaler()
    X = X.copy()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    # Split stratifié train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        stratify=y
    )
    print(f"🔀 Split — Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # SMOTE pour équilibrer les classes (uniquement sur le train)
    smote = SMOTE(random_state=params["split"]["random_state"])
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(
        f"⚖️  Après SMOTE — fraudes: {y_train_res.sum()} | "
        f"non-fraudes: {(y_train_res == 0).sum()}"
    )

    # Sauvegarder les données processées
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pd.concat([X_train_res, y_train_res], axis=1).to_csv("data/processed/train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv("data/processed/test.csv", index=False)
    joblib.dump(scaler, "data/processed/scaler.pkl")

    print("✅ Preprocessing terminé — fichiers sauvegardés dans data/processed/")


if __name__ == "__main__":
    preprocess()
