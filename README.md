# Intelligent System Development Assignment 02

From Data Representation to a Deployable Intelligent System. This repository contains three reproducible machine-learning applications, a FastAPI backend, a responsive React web client, and an Expo mobile client. The released web application runs on Vercel and calls the released backend on Render.

## Three applications

1. **Diabetes Prediction** — binary classification from six raw patient attributes; persisted Random Forest Pipeline.
2. **Vietnam House Price Prediction** — regression from 11 mixed property attributes; persisted Random Forest Pipeline.
3. **E-commerce Customer Preference** — rating-derived binary preference classification from review text and helpfulness values; persisted TF-IDF + Logistic Regression Pipeline.

## Dataset sources

- Diabetes: [Kaggle Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), local `data/diabetes/diabetes.csv` (768 rows). The original A1 notebook recorded Kaggle but did not preserve its dataset-card URL; this widely used card matches the local schema.
- House price: [Kaggle Vietnam Housing Dataset 2024](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024), local `data/house_price/vietnam_housing_dataset.csv` (30,229 rows).
- E-commerce: [Kaggle Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews), actual local file `data/ecommerce/Reviews.csv` (568,454 rows, 10 columns).

The 300+ MB Ecommerce CSV is intentionally excluded from Git. To reproduce the notebook, download `Reviews.csv` from the Kaggle link above and place it at exactly `data/ecommerce/Reviews.csv`. The deployed backend does not need this training CSV because it loads the persisted fitted Pipeline.

## Representation and results

| Application | Raw representation | Final numerical representation | Saved model | Held-out result |
|---|---|---|---|---|
| Diabetes | 6 CSV values | median-imputed, scaled `B × 6` | Random Forest Classifier | Accuracy 0.7468; F1 0.6061 |
| House | 11 mixed values | imputed/scaled/one-hot `B × 83` | Random Forest Regressor | MAE 1.2529; RMSE 1.6018; R² 0.4738 |
| E-commerce | Summary, Text, 2 helpfulness counts | TF-IDF `V=12,000` + 5 engineered values = `B × 12,005` | Logistic Regression | Accuracy 0.9566; F1 0.9746; ROC-AUC 0.9842 |

E-commerce uses Score only to derive `Score 4–5 → Positive`, `Score 1–2 → Negative`; neutral Score 3 is removed. Score, target, Id, ProductId, UserId and ProfileName are excluded from model input. Of 568,454 source rows, 524,983 are usable after neutral, invalid-helpfulness and duplicate-review removal. A reproducible stratified 120,000-row subset is split 70/15/15.

## Repository structure

```text
data/                    Three local datasets
notebooks/               Three final notebooks plus experiments/
models/                  Three fitted joblib Pipelines and Ecommerce evidence
figures/                 Evaluation/representation figures by application
shared_ml/               Importable Ecommerce custom transformers
scripts/                 Reproducible training/notebook/audit utilities
backend/                 FastAPI, schemas, metadata, services and pytest
web/                     React/Vite/TypeScript responsive application
mobile/                  Expo React Native/TypeScript demonstration
knowledge_graph/         Optional Diabetes Neo4j bonus
docs/                    API, architecture, final A2 report and archived A1 report
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the exact final tree and [docs/ASSIGNMENT_02_COMPARISON.md](docs/ASSIGNMENT_02_COMPARISON.md) for the cross-application discussion.

## Final Assignment 02 report

- [Editable report source](docs/report_assignment_02/ASSIGNMENT_02_REPORT.md)
- [Microsoft Word report](docs/report_assignment_02/ASSIGNMENT_02_REPORT.docx)
- [PDF report](docs/report_assignment_02/ASSIGNMENT_02_REPORT.pdf)

The report is written in Vietnamese, uses the executed notebook evidence and persisted results, and includes real production Web and Expo screenshots. Rebuild the DOCX and report assets with `python docs/report_assignment_02/build_report.py`.

## Setup and notebook execution

Python 3.12 and Node.js 20+ are recommended. The saved artifacts use scikit-learn 1.9.0.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute notebooks/ecommerce/03_ecommerce_interest_system.ipynb --inplace --ExecutePreprocessor.timeout=600
```

The Ecommerce experiment can be reproduced with `python -m scripts.train_ecommerce`. It keeps TF-IDF sparse, uses `random_state=42`, records A/B/C representation results and six-model validation timings, persists the full Pipeline, and verifies fresh-load predictions. Diabetes and House preserve their valid 80/20 plus training-only CV methodology.

## Backend and API

```powershell
python -m pip install -r backend\requirements.txt
$env:PYTHONPATH = "backend;."
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Render-compatible native build/start configuration from the repository root:

```sh
pip install -r backend/requirements.txt
PYTHONPATH=backend:. uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
```

Alternatively, `backend/Dockerfile` copies `backend/`, all three `models/`, and `shared_ml/` into the Linux image. Configure production `CORS_ORIGINS`, model/Neo4j settings, and secrets in the hosting platform rather than committing an `.env` file.

Endpoints:

- `GET /health`
- `GET /api/v1/models`
- `GET /api/v1/models/{diabetes|house-price|ecommerce}`
- `POST /api/v1/diabetes/predict`
- `POST /api/v1/house-price/predict`
- `POST /api/v1/ecommerce/predict`
- `GET /api/v1/diabetes/knowledge-graph` (optional bonus; Neo4j failure is non-fatal)

Swagger UI is at `http://localhost:8000/docs`. Full payloads are in [docs/API.md](docs/API.md).

## Web

Production: [https://intelligent-system-assignment-01.vercel.app](https://intelligent-system-assignment-01.vercel.app)

```powershell
Set-Location web
npm install
npm run dev
npm run lint
npm run build
```

Set `VITE_API_BASE_URL`. Routes are `/`, `/diabetes`, `/house-price`, `/ecommerce`, `/diabetes/knowledge-graph`, and `/about`. Responsive rules cover 360, 390, 412, 768 and 1024+ px layouts.

## Mobile

```powershell
Set-Location mobile
npm install
npm run typecheck
npx expo start
npx expo export --platform android
```

Set `EXPO_PUBLIC_API_BASE_URL`; no hosted URL is hard-coded. Primary screens are Home, Diabetes, House Price, E-commerce and About.

## Tests and persisted models

```powershell
$env:PYTHONPATH = "backend;."
.\.venv\Scripts\python.exe -m pytest
```

Models live at:

- `models/diabetes/diabetes_model.joblib`
- `models/house_price/house_price_model.joblib`
- `models/ecommerce/ecommerce_interest_model.joblib`

Tests validate contracts, invalid payloads, direct-versus-HTTP equality, fresh loading, absence of inference-time `fit`, and independence from Neo4j.

## Reproducibility, deployment status, and limitations

Preprocessors are fitted only on training data; held-out test data is not used for tuning. Exact duplicate Ecommerce reviews are removed before splitting. The full fitted transformations and estimators are serialized together.

Production backend: [https://intelligent-system-assignment-01.onrender.com](https://intelligent-system-assignment-01.onrender.com). Production web: [https://intelligent-system-assignment-01.vercel.app](https://intelligent-system-assignment-01.vercel.app). The mobile application remains an Expo demonstration and is not published as a store application.

Diabetes is not diagnosis; House is not professional valuation; Ecommerce predicts a rating-derived proxy rather than independently observed intention. Class imbalance, listing-price bias, small medical data, and near-duplicate/user/product dependencies remain limitations.
