# Assignment 02 REST API

Base URL is configured by the client; examples use `http://localhost:8000`. JSON payloads reject unknown fields and non-finite values. The backend performs no fitting or manual reimplementation of preprocessing.

## Discovery and health

- `GET /health` reports `diabetes`, `house_price`, `ecommerce`, and optional `neo4j` status.
- `GET /api/v1/models` returns all three metadata records.
- `GET /api/v1/models/diabetes`
- `GET /api/v1/models/house-price`
- `GET /api/v1/models/ecommerce`

## Diabetes

`POST /api/v1/diabetes/predict`

```json
{"Pregnancies":1,"Glucose":85,"BloodPressure":66,"BMI":24.0,"DiabetesPedigreeFunction":0.2,"Age":23}
```

Response includes integer `prediction`, educational `label`, positive-class `probability`, model name and disclaimer. This is not diagnosis.

## Vietnam house price

`POST /api/v1/house-price/predict`

```json
{"Province":"Hồ Chí Minh","Area":45,"Frontage":4,"Access Road":4,"House direction":"Đông - Nam","Balcony direction":"Đông - Nam","Floors":3,"Bedrooms":3,"Bathrooms":3,"Legal status":"Have certificate","Furniture state":"Full"}
```

Nullable raw fields may be `null` where metadata permits; the saved Pipeline imputes them. Response is an educational estimate in billion VND, not professional valuation.

## E-commerce customer preference

`POST /api/v1/ecommerce/predict`

```json
{"Summary":"Excellent","Text":"Fresh, tasty and exactly as described. I would buy it again.","HelpfulnessNumerator":1,"HelpfulnessDenominator":1}
```

Contract rules:

- `Text` must be nonblank and no longer than 20,000 characters.
- `Summary` is a string no longer than 500 characters and may be empty.
- Helpfulness values are nonnegative integers and numerator cannot exceed denominator.
- Score, target and identifiers are forbidden.

Response:

```json
{"prediction":1,"label":"Positive","probability":0.99,"model":"Logistic Regression","disclaimer":"..."}
```

The exact probability varies with input. It is the fitted Logistic Regression positive-class probability. The label is **predicted preference**, not actual intention.

## Knowledge graph bonus and errors

`GET /api/v1/diabetes/knowledge-graph` remains optional. If Neo4j is unavailable it returns 503 without affecting any prediction endpoint. Validation errors return 422, oversized bodies 413, and safe inference failures 500.
