# Intelligent System Development Assignment 01

Local end-to-end application for two fitted machine-learning systems. The application does **not** train models: FastAPI loads the trusted local scikit-learn Pipelines in `models/` once at startup, validates their raw feature contracts, and uses them for every prediction from the web and mobile clients.

This repository is built for local demonstration only. Nothing in this project is deployed.

## Systems

### Diabetes Classification

- Dataset: Kaggle diabetes dataset used in the notebook, 768 observations.
- Final model: Random Forest Classifier (`n_estimators=100`, `max_depth=6`, `random_state=42`).
- Exact raw features: `Pregnancies`, `Glucose`, `BloodPressure`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- Fitted preprocessing: median imputation and standard scaling inside the saved Pipeline.
- Held-out metrics: Accuracy 0.7468, Precision 0.6667, Recall 0.5556, F1 0.6061.
- Limitation: an educational classification demonstration, not a medical diagnosis or clinical decision system.

### Vietnam House Price Regression

- Dataset: 30,229 Vietnam housing listings in the cleaned notebook representation; the source data is the 2024 collection documented by the notebook.
- Final model: Random Forest Regressor (`n_estimators=100`, `max_depth=12`, `random_state=42`).
- Exact **11 raw features**: `Province`, `Area`, `Frontage`, `Access Road`, `House direction`, `Balcony direction`, `Floors`, `Bedrooms`, `Bathrooms`, `Legal status`, `Furniture state`.
- Fitted preprocessing: numerical median imputation/scaling and categorical most-frequent imputation/one-hot encoding with unknown categories ignored.
- Target: original `Price` in billion VND.
- Held-out metrics: MAE 1.2529 billion VND, RMSE 1.6018 billion VND, R² 0.4738, MAPE 27.36%.
- Limitation: listed prices are not necessarily transactions; the model is an educational estimate, not a professional valuation.

## Architecture

```text
ML Notebooks
      ↓
Saved fitted sklearn Pipelines
      ↓
FastAPI ─────────────→ Neo4j Diabetes Knowledge Graph
      ↓
Web / Expo Mobile
      ↓
User-facing educational prediction
```

The training Pipeline equals the inference Pipeline. API code creates a one-row DataFrame with the exact saved raw column names/order, then calls `predict` (and `predict_proba` for Diabetes). It never refits, scales, imputes, or encodes separately. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Project Structure

```text
backend/             FastAPI app, metadata, services, schemas and pytest tests
web/                 React + Vite + TypeScript responsive web app
mobile/              Expo React Native + TypeScript mobile app
knowledge_graph/     Idempotent Neo4j Cypher and graph documentation
docs/                API contract and architecture documentation
models/              Trusted fitted joblib Pipelines
notebooks/           Final ML notebooks (unchanged by application inference)
data/                Local assignment datasets
figures/             Notebook output figures
docker-compose.yml   Neo4j, backend, seed helper and web services
```

## Prerequisites

- Python 3.12 (the saved models were produced with Python 3.12 and scikit-learn 1.9.0)
- Node.js 20+ and npm
- Docker Desktop with Docker Compose for the Neo4j integration
- The two required model files under `models/diabetes/` and `models/house_price/`

## Run Backend

PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
$env:PYTHONPATH = "backend"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Startup fails clearly when a model/metadata file is missing or its feature contract disagrees with `feature_names_in_`. Neo4j connectivity failure is non-fatal; `/health` reports it while both prediction APIs remain operational.

Swagger UI: <http://localhost:8000/docs>

Linux/cloud start command from the repository root (the platform supplies `PORT`):

```sh
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
```

Install backend runtime dependencies from `backend/requirements.txt`. Configure
`NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in the hosting environment; no
cloud credentials belong in the repository.

## Run Neo4j and Seed the Graph

Copy `.env.example` to `.env`, change the example local password, then use the same value for local backend configuration.

```powershell
docker compose up -d neo4j
docker compose --profile tools run --rm neo4j-seed
```

Neo4j Browser is at <http://localhost:7474>, Bolt is at `bolt://localhost:7687`. The seed is idempotent because it uses `MERGE`.

To run backend + Neo4j + web entirely with Docker:

```powershell
docker compose up --build neo4j backend web
```

The Docker web UI is then at <http://localhost:8080>.

## Run Web

```powershell
Copy-Item web\.env.example web\.env
Set-Location web
npm install
npm run dev
```

Open <http://localhost:5173>. Production-style local validation uses `npm run lint` and `npm run build`. `VITE_API_BASE_URL` controls the backend URL.

## Run Mobile

```powershell
Copy-Item mobile\.env.example mobile\.env
Set-Location mobile
npm install
npx expo start
```

Configure `EXPO_PUBLIC_API_BASE_URL` for the runtime target:

- Browser and usually iOS simulator: `http://localhost:8000`
- Android emulator: commonly `http://10.0.2.2:8000`
- Physical phone: the computer's LAN IP such as `http://192.168.x.x:8000`

The phone and computer must be reachable on the same network, and the backend must listen on `0.0.0.0`. The repository does not guess or commit a LAN IP.

## Run Tests

```powershell
$env:PYTHONPATH = "backend"
.\.venv\Scripts\python.exe -m pytest

Set-Location web
npm run lint
npm run build

Set-Location ..\mobile
npm run typecheck
```

Tests compare notebook demo requests through the API against direct predictions from the saved joblib models, verify exact feature contracts, invalid payload handling, and assert that prediction never calls `fit`.

## API Docs

The complete shared contract and exact examples are in [docs/API.md](docs/API.md). Main endpoints:

- `GET /health`
- `GET /api/v1/models`
- `GET /api/v1/models/diabetes`
- `GET /api/v1/models/house-price`
- `POST /api/v1/diabetes/predict`
- `POST /api/v1/house-price/predict`
- `GET /api/v1/diabetes/knowledge-graph`

## Important Notes

- Only trusted local `.joblib` files are loaded; uploads and arbitrary pickle execution are not accepted.
- Request payloads are schema-validated, extra fields are rejected, non-finite values are rejected, and request size is limited.
- CORS is configured through comma-separated `CORS_ORIGINS`; no production wildcard is the default.
- No real password is committed. Change example credentials for local use.
- Diabetes output is labelled **Prediction**, never diagnosis. House output is an educational estimate, never appraisal or investment advice.
- This task intentionally performs no cloud deployment, Expo publishing, Docker Hub publishing, Git commit, or Git push.
