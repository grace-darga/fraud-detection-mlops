# 🔍 Fraud Detection MLOps

Système de détection de fraude bancaire en temps réel avec pipeline MLOps complet.

## Stack technologique

| Composant | Technologie |
|-----------|-------------|
| Modèle ML | XGBoost + SMOTE |
| Tracking | MLflow |
| Versioning données | DVC |
| API | FastAPI |
| Conteneurisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Tests | pytest |

## Architecture du projet

```
fraud-detection-mlops/
├── data/
│   ├── raw/                  # Données brutes (géré par DVC)
│   └── processed/            # Données transformées
├── src/
│   ├── data/
│   │   ├── download.py       # Téléchargement dataset Kaggle
│   │   └── preprocess.py     # Preprocessing + SMOTE
│   ├── models/
│   │   ├── train.py          # Entraînement XGBoost + MLflow
│   │   └── evaluate.py       # Métriques complètes
│   └── api/
│       ├── main.py           # FastAPI app
│       └── schemas.py        # Pydantic models
├── tests/
│   ├── test_api.py           # Tests API
│   └── test_model.py         # Tests modèle/données
├── .github/
│   └── workflows/
│       └── ci_cd.yml         # GitHub Actions pipeline
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                  # Pipeline DVC reproductible
├── params.yaml               # Hyperparamètres
└── requirements.txt
```

## Démarrage rapide

### 1. Installation

```bash
# Cloner et créer l'environnement
git clone <your-repo>
cd fraud-detection-mlops
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Initialiser Git + DVC

```bash
git init
dvc init
git add .
git commit -m "init: project structure"
```

### 3. Télécharger les données

Configurez vos credentials Kaggle (`~/.kaggle/kaggle.json`) puis :

```bash
python src/data/download.py
```

### 4. Lancer le pipeline complet

```bash
# Via DVC (reproductible)
dvc repro

# Ou étape par étape
python src/data/preprocess.py
python src/models/train.py
python src/models/evaluate.py
```

### 5. Suivre les expériences MLflow

```bash
mlflow ui
# → http://localhost:5000
```

### 6. Lancer l'API

```bash
# Développement
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs

# Production via Docker
docker-compose up --build
```

### 7. Tester l'API

```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0,
    "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.02, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02
  }'
```

### 8. Lancer les tests

```bash
pytest tests/ -v
```

## Pipeline DVC

```
data/raw/creditcard.csv
    │
    ▼ preprocess
data/processed/ (train.csv, test.csv, scaler.pkl)
    │
    ▼ train
models/model.pkl + metrics/scores.json
    │
    ▼ evaluate
metrics/full_report.json
```

Commandes DVC utiles :

```bash
dvc repro                # Reproduire le pipeline
dvc metrics show         # Afficher les métriques
dvc metrics diff         # Comparer avec le commit précédent
dvc dag                  # Visualiser le DAG
dvc push                 # Sauvegarder vers le remote
dvc pull                 # Récupérer depuis le remote
```

## CI/CD Pipeline

Le pipeline GitHub Actions s'exécute sur chaque push :

1. **Tests & Linting** — pytest + flake8
2. **Train & Evaluate** (main uniquement) — DVC pipeline + validation des seuils
3. **Build & Deploy** (main uniquement) — Docker Hub + déploiement VPS

### Secrets GitHub requis

| Secret | Description |
|--------|-------------|
| `GDRIVE_FOLDER_ID` | ID du dossier Google Drive pour DVC |
| `GDRIVE_CREDENTIALS` | JSON des credentials service account |
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN` | Token d'accès Docker Hub |
| `VPS_HOST` | IP du serveur de production |
| `VPS_USER` | Utilisateur SSH du serveur |

## Hyperparamètres

Modifiez `params.yaml` pour changer les hyperparamètres :

```yaml
model:
  n_estimators: 200    # Nombre d'arbres
  max_depth: 6         # Profondeur max des arbres
  learning_rate: 0.1   # Taux d'apprentissage
  scale_pos_weight: 10 # Poids des fraudes (déséquilibre des classes)
  threshold: 0.5       # Seuil de décision
```

Puis relancez : `dvc repro`

## Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Page d'accueil |
| GET | `/health` | Santé de l'API |
| GET | `/docs` | Documentation Swagger |
| POST | `/predict` | Prédiction unique |
| POST | `/predict/batch` | Prédictions en lot (max 100) |

## Performances attendues

| Métrique | Seuil minimum | Attendu |
|----------|---------------|---------|
| ROC-AUC | 0.95 | ~0.98 |
| F1-Score | 0.80 | ~0.85 |
| Precision | — | ~0.90 |
| Recall | — | ~0.80 |

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle

- 284 807 transactions sur 2 jours
- 492 fraudes (0.172%)
- Features V1-V28 : résultats d'une PCA (anonymisées)
- Features originales : Time (secondes) et Amount (euros)
