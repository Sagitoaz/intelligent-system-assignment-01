# Shared API Contract

Base URL for local development: `http://localhost:8000`. Interactive OpenAPI documentation: <http://localhost:8000/docs>.

All prediction requests use `Content-Type: application/json`. Field names deliberately match the saved Pipeline's raw `feature_names_in_` values exactly. Unknown extra keys are rejected with 422.

## Health

`GET /health`

```json
{
  "status": "ok",
  "models": { "diabetes": "loaded", "house_price": "loaded" },
  "neo4j": "available"
}
```

`neo4j` can be `available`, `unavailable`, or `disabled`. Model predictions remain operational when it is unavailable.

## Model metadata

- `GET /api/v1/models`
- `GET /api/v1/models/diabetes`
- `GET /api/v1/models/house-price`

Metadata includes task, expected raw field order, data types, descriptions, units, nullability, fitted known categorical options, model configuration, metrics, target/class labels, and disclaimer. Both clients use it to render forms.

## Diabetes prediction

`POST /api/v1/diabetes/predict`

Exact request schema and order:

1. `Pregnancies` — non-negative integer
2. `Glucose` — positive finite number
3. `BloodPressure` — positive finite number
4. `BMI` — positive finite number
5. `DiabetesPedigreeFunction` — non-negative finite number
6. `Age` — positive integer

Notebook demo request:

```json
{
  "Pregnancies": 1,
  "Glucose": 85,
  "BloodPressure": 66,
  "BMI": 24.0,
  "DiabetesPedigreeFunction": 0.2,
  "Age": 23
}
```

Response from the saved model (probability shown here rounded for documentation):

```json
{
  "prediction": 0,
  "label": "Non-diabetic",
  "probability": 0.01054453459068126,
  "model": "Random Forest Classifier",
  "disclaimer": "This prediction is for educational demonstration only and is not a medical diagnosis."
}
```

## House-price prediction

`POST /api/v1/house-price/predict`

Exact request schema/order: `Province`, `Area`, `Frontage`, `Access Road`, `House direction`, `Balcony direction`, `Floors`, `Bedrooms`, `Bathrooms`, `Legal status`, `Furniture state`.

All 11 keys are required. `Province` and `Area` require values. The other house attributes may be explicit JSON `null` when unavailable; the fitted Pipeline performs its own learned imputation. Known options are served by metadata. An unknown nonblank category is accepted because the fitted encoder uses `handle_unknown="ignore"`.

Notebook demo request:

```json
{
  "Province": "Hồ Chí Minh",
  "Area": 45.0,
  "Frontage": 4.0,
  "Access Road": 4.0,
  "House direction": "Đông - Nam",
  "Balcony direction": "Đông - Nam",
  "Floors": 3.0,
  "Bedrooms": 3.0,
  "Bathrooms": 3.0,
  "Legal status": "Have certificate",
  "Furniture state": "Full"
}
```

Response (full floating-point value is returned; `formatted` is rounded for display):

```json
{
  "predicted_price": 5.126661845433643,
  "unit": "billion VND",
  "formatted": "5.13 billion VND",
  "model": "Random Forest Regressor",
  "disclaimer": "Educational estimate only; not a professional property valuation."
}
```

## Diabetes Knowledge Graph

`GET /api/v1/diabetes/knowledge-graph`

```json
{
  "nodes": [
    {
      "id": "neo4j-element-id",
      "labels": ["AssignmentEntity", "Model"],
      "properties": { "name": "Random Forest Classifier", "task": "Classification" }
    }
  ],
  "edges": [
    {
      "id": "neo4j-relationship-id",
      "source": "source-element-id",
      "target": "target-element-id",
      "type": "USES_FEATURE",
      "properties": { "importance": 0.40171561704467146 }
    }
  ]
}
```

Returns HTTP 503 with a clear `detail` when Neo4j is unavailable.

## Errors

- `413` request exceeds configured size.
- `422` missing/extra field, wrong type, blank categorical value, invalid sign, NaN, or infinity.
- `500` unexpected internal model failure; no raw traceback in the response.
- `503` Neo4j unavailable for graph reads.
