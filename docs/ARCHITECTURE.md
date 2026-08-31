# Assignment 02 Architecture

## End-to-end flow

```text
Raw CSV/text → understand/clean → fitted numerical representation
             → train/validate/test → complete joblib Pipeline
             → FastAPI (load once) → React web / Expo mobile
                         └──────────→ optional Diabetes Neo4j graph
```

The core invariant is **training Pipeline = inference Pipeline**. FastAPI validates raw JSON, creates a one-row DataFrame in metadata order, then calls `predict`/`predict_proba`. It never fits a vectorizer, scaler, encoder, imputer, transformer or estimator.

## Three representations

- Diabetes: six numeric values → hidden-zero cleaning → median imputation → standard scaling → `B × 6`.
- House: 11 mixed fields → numeric median/scaling and categorical mode/one-hot → `B × 83`.
- Ecommerce: Summary + Text → sparse TF-IDF (`V=12,000`) and four raw values → five engineered/scaled numerical features → sparse `B × 12,005`.

`shared_ml/ecommerce_transformers.py` makes custom feature logic importable in notebooks, FastAPI, tests and fresh processes. The saved Ecommerce artifact contains both transformer branches and Logistic Regression. Lambda/notebook-local pickle dependencies are avoided.

## Leakage boundaries

Diabetes and House keep the existing 80/20 held-out test plus training-only CV. Ecommerce removes exact duplicate review keys before a stratified 70/15/15 split. Score creates y and is immediately excluded; identifiers and target-derived features are excluded. TF-IDF vocabulary and scaling are fitted on train only during comparison, then the frozen configuration is refitted on train+validation before one test evaluation.

## Runtime components

One FastAPI process loads three trusted local artifacts at startup and verifies metadata against each `feature_names_in_`. Neo4j connectivity is optional/non-fatal. Web and mobile discover raw form fields from metadata and send no transformed values. `VITE_API_BASE_URL` and `EXPO_PUBLIC_API_BASE_URL` control connectivity.

The Docker backend copies `shared_ml`, models and backend code and exposes port 8000. The responsive Vite client is the primary polished UI; Expo is a minimal functional demonstration.

## Security and failure handling

Only repository-owned joblib files are loaded. Extra payload fields, blanks, invalid ranges, non-finite values and oversized bodies are rejected. Error responses do not echo invalid sensitive values. Prediction services remain available when the knowledge graph is offline.
