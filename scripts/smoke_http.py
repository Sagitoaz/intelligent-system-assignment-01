import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

base = os.getenv("SMOKE_API_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
sys.stdout.reconfigure(encoding="utf-8")

def call(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(base + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as response:
            value = json.load(response)
            print(path, response.status, "PASS")
            return value
    except HTTPError as exc:
        print(path, exc.code, "EXPECTED_OPTIONAL_FAILURE" if path.endswith("knowledge-graph") else "FAIL")

call("/health")
call("/api/v1/models")
call("/api/v1/models/diabetes")
call("/api/v1/models/house-price")
call("/api/v1/models/ecommerce")
call("/api/v1/diabetes/predict", {"Pregnancies":1,"Glucose":85,"BloodPressure":66,"BMI":24,"DiabetesPedigreeFunction":0.2,"Age":23})
call("/api/v1/house-price/predict", {"Province":"Hồ Chí Minh","Area":45,"Frontage":4,"Access Road":4,"House direction":"Đông - Nam","Balcony direction":"Đông - Nam","Floors":3,"Bedrooms":3,"Bathrooms":3,"Legal status":"Have certificate","Furniture state":"Full"})
call("/api/v1/ecommerce/predict", {"Summary":"Excellent","Text":"Fresh, tasty and exactly as described. I would buy it again.","HelpfulnessNumerator":1,"HelpfulnessDenominator":1})
call("/api/v1/diabetes/knowledge-graph")
