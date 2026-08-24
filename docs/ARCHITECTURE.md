# Application Architecture

## Runtime topology

```text
User
│
├── React/Vite Web
│
└── Expo React Native Mobile
          │
          ▼
       FastAPI
          │
          ├── ModelService: Diabetes
          │      └── models/diabetes/diabetes_model.joblib
          │              └── median imputer → scaler → Random Forest classifier
          │
          ├── ModelService: House Price
          │      └── models/house_price/house_price_model.joblib
          │              └── ColumnTransformer → Random Forest regressor
          │
          └── Neo4jService
                 └── Neo4j Diabetes model/provenance graph
```

FastAPI model loading is part of application lifespan. Missing models or mismatched contracts stop startup; Neo4j failure is logged and represented as `unavailable` but does not prevent the model service from starting.

## Representation consistency

```text
User raw values
  → Pydantic type/finite/basic-sign validation
  → one-row pandas DataFrame in exact feature_names_in_ order
  → saved fitted Pipeline
      → learned imputation
      → learned scaling / one-hot encoding
      → fitted estimator
  → response formatting
```

The central invariant is:

```text
Training Pipeline = Inference Pipeline
```

The application does not reproduce preprocessing. It does not call `fit`, `fit_transform`, `StandardScaler`, `SimpleImputer`, or `OneHotEncoder` during prediction. Converting an explicit JSON `null` to `numpy.nan` for nullable house fields only represents a missing raw value so the saved Pipeline can perform its learned imputation.

## Model contracts

JSON metadata under `backend/app/model_metadata/` is notebook-derived and checked against the saved objects. Startup checks:

1. metadata field order equals `expected_raw_features`;
2. expected features equal the Pipeline `feature_names_in_` exactly, including case and spaces;
3. the saved object is a prediction Pipeline;
4. House numerical/categorical groups equal the fitted `ColumnTransformer` columns;
5. House UI category options equal fitted encoder categories;
6. the encoder's unknown category policy is `ignore`.

Web and mobile fetch this metadata to render forms, keeping all clients aligned with the backend contract.

## Reliability boundaries

- Models load once at startup, never once per request.
- Prediction failures return a generic 500 response without a raw traceback.
- Pydantic returns structured 422 validation errors for missing, extra, wrong-type, blank, non-positive where applicable, NaN, or infinite inputs.
- Payloads larger than `MAX_REQUEST_BYTES` are rejected.
- Neo4j graph reads return 503 when unavailable; prediction endpoints remain independent.
- Only repository-controlled model paths resolved from `PROJECT_ROOT` are loaded.

## Knowledge graph

The graph is transparent and model-centric, not a graph of unsourced medical claims. Its nodes describe the assignment system, selected classifier, six raw features, Outcome target, fitted pipeline steps, dataset provenance, model-selection experiment, and held-out metrics. Exact fitted feature importances live on `USES_FEATURE` relationships and are explicitly labelled impurity-based rather than causal.
