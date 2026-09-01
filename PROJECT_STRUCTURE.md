# Intelligent System Development Assignment 02 — Repository Structure

## Overview

The repository now implements three end-to-end intelligent systems: Diabetes classification, Vietnam House Price regression, and E-commerce Customer Preference text classification. Each follows raw data → cleaning → numerical representation → learning/evaluation → persisted fitted Pipeline → FastAPI → web/mobile inference. Neo4j remains an optional Diabetes transparency bonus.

## Actual repository tree

Generated caches, virtual environments, `node_modules` and build output are omitted.

```text
├── data/
│   ├── diabetes/diabetes.csv
│   ├── house_price/vietnam_housing_dataset.csv
│   └── ecommerce/Reviews.csv
├── notebooks/
│   ├── diabetes/01_diabetes_system.ipynb
│   ├── house_price/02_house_price_system.ipynb
│   ├── ecommerce/03_ecommerce_interest_system.ipynb
│   └── experiments/tfidf_5_sentences.ipynb
├── models/
│   ├── diabetes/diabetes_model.joblib
│   ├── house_price/house_price_model.joblib
│   └── ecommerce/
│       ├── ecommerce_interest_model.joblib
│       ├── training_results.json
│       └── error_examples.csv
├── figures/
│   ├── diabetes/
│   ├── house_price/
│   ├── ecommerce/
│   └── png/
│       ├── web/                     # Home and three production prediction pages
│       └── mobile/                  # Expo Home/forms/results for three systems
├── shared_ml/
│   ├── __init__.py
│   └── ecommerce_transformers.py
├── scripts/
│   ├── audit_assignment02.py
│   ├── add_house_a2_models.py
│   ├── build_assignment02_notebooks.py
│   ├── refresh_ecommerce_artifact.py
│   ├── release_file_audit.py
│   ├── smoke_http.py
│   ├── smoke_models.py
│   ├── validate_notebooks.py
│   └── train_ecommerce.py
├── backend/
│   ├── app/
│   │   ├── api/routes.py
│   │   ├── core/config.py
│   │   ├── model_metadata/{diabetes,house_price,ecommerce}.json
│   │   ├── schemas/{predictions,graph}.py
│   │   ├── services/{model_service,neo4j_service}.py
│   │   └── main.py
│   ├── tests/{conftest,test_api,test_contract}.py
│   ├── Dockerfile
│   └── requirements.txt
├── web/
│   ├── vercel.json                 # SPA fallback for direct route refresh
│   └── src/
│       ├── pages/{HomePage,DiabetesPage,HousePricePage,EcommercePage,KnowledgeGraphPage,AboutPage}.tsx
│       ├── components/{Layout,PredictionForm}.tsx
│       └── {api,types,numericValidation,styles}.ts(x)
├── mobile/
│   └── src/
│       ├── screens/{HomeScreen,DiabetesScreen,HousePriceScreen,EcommerceScreen,AboutScreen,KnowledgeGraphScreen}.tsx
│       ├── components/PredictionScreen.tsx
│       └── {api,types,theme}.ts
├── knowledge_graph/
│   ├── cypher/diabetes_knowledge_graph.cypher
│   └── README.md
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── ASSIGNMENT_02_COMPARISON.md
│   ├── report_assignment_02/
│   │   ├── ASSIGNMENT_02_REPORT.{md,docx,pdf}
│   │   ├── build_report.py
│   │   └── assets/                  # generated architecture and screenshot composites
│   └── archive/assignment_01/       # former A1 report, PDF, assets and builder
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── pytest.ini
├── docker-compose.yml
└── .env.example
```

## Data, notebooks, models, figures, and shared ML

The three `data/` folders preserve source CSVs. The three deliverable notebooks contain executed inspection, representation and evaluation evidence; the original toy TF-IDF notebook is clearly separated under `experiments/`. Models are full fitted Pipelines. Ecommerce-specific custom transformers are importable from `shared_ml`, so a fresh joblib load does not depend on notebook state. Figures are grouped by application; Ecommerce includes score/target/length/helpfulness, terms, representation/model comparisons, confusion matrix and ROC.

## Backend, metadata, and endpoints

One FastAPI lifespan loads all three models once. Metadata defines the exact raw contracts and is verified against persisted `feature_names_in_`. Endpoints are health, model list/three detail routes, three prediction routes and the optional Diabetes graph route; see `docs/API.md`.

## Web pages and mobile screens

The responsive web routes are Home, Diabetes, House Price, E-commerce, optional Knowledge Graph and About. Forms are metadata-driven and show scientifically accurate result language. Mobile primary navigation contains Home plus the three prediction systems and About; the old graph implementation remains in source as an optional secondary bonus but is removed from primary navigation.

## Training and inference workflows

Diabetes and House preserve valid A1 80/20 + training-only CV and their saved Random Forest Pipelines; House adds Ridge and Gradient Boosting training-fold benchmarks without test-driven reselection. Ecommerce uses deterministic cleaning, stratified 120,000-row sampling, 70/15/15 splits, controlled tabular/text/combined comparison, six sparse-compatible models, a validation-only selection rule, one final test evaluation, and complete artifact persistence.

At inference, web/mobile raw JSON → Pydantic validation → ordered one-row DataFrame → saved Pipeline preprocessing/prediction → response. No inference-time fitting occurs.

## Configuration, tests, and archive

`VITE_API_BASE_URL`, `EXPO_PUBLIC_API_BASE_URL`, Neo4j variables, CORS, and request-size settings are documented in example environment files. Vercel uses `web/vercel.json` for SPA route fallback. Pytest covers the three model contracts and predictions; npm scripts validate both clients. The large local Ecommerce CSV, environment files, dependencies, caches and build output are excluded by `.gitignore`, while the three deployment model artifacts remain candidates for Git. The complete Assignment 01 report tree was moved to `docs/archive/assignment_01/` and is not presented as the current report. The final Assignment 02 report is maintained as Markdown, editable DOCX, and a 13-page A4 PDF under `docs/report_assignment_02/`; its builder also regenerates the architecture and screenshot composites from repository evidence.
