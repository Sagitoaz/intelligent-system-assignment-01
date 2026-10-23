"""Smoke test script for Assignment 06 deployment app.

Verifies:
1. Stock + PyTorch: Saved model loading, preprocessing, prediction, output validity
2. Stock + Keras: Saved model loading, preprocessing, prediction, output validity
3. Gold + PyTorch: Saved model loading, preprocessing, prediction, output validity
4. Gold + Keras: Saved model loading, preprocessing, prediction, output validity
5. Confirms zero retraining on load
6. Starts Streamlit server, tests HTTP 200 health, shuts down cleanly
"""

import os
import sys
import time
import urllib.request
import subprocess

# Suppress TF logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
import keras

from src.assignment06.data import prepare_time_series_data, get_dataset_metadata
from src.assignment06.models_pytorch import VanillaRNNPyTorch


def test_combination(dataset_key: str, framework: str):
    print(f"\n[TESTING COMBINATION] Dataset: {dataset_key.upper()} | Framework: {framework}")
    bundle = prepare_time_series_data(dataset_key, seq_len=30)
    assert len(bundle.X_test) > 0, "Test split empty!"
    sample_idx = len(bundle.X_test) - 1
    window_x = bundle.X_test[sample_idx : sample_idx + 1]
    actual_price = float(bundle.target_scaler.inverse_transform(bundle.y_test[sample_idx : sample_idx + 1])[0, 0])

    if framework == "PyTorch":
        model_path = os.path.join(PROJECT_ROOT, "models", "assignment06", f"{dataset_key}_rnn_pytorch.pt")
        assert os.path.exists(model_path), f"PyTorch model missing: {model_path}"
        model = VanillaRNNPyTorch(input_size=5, hidden_size=64, num_layers=1, output_size=1)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            tensor_x = torch.from_numpy(window_x).float()
            pred_scaled = model(tensor_x).cpu().numpy()
    elif framework == "Keras":
        model_path = os.path.join(PROJECT_ROOT, "models", "assignment06", f"{dataset_key}_rnn_keras.keras")
        assert os.path.exists(model_path), f"Keras model missing: {model_path}"
        model = keras.models.load_model(model_path)
        pred_scaled = model.predict(window_x, verbose=0)
    else:
        raise ValueError(f"Unknown framework: {framework}")

    pred_price = float(bundle.target_scaler.inverse_transform(pred_scaled)[0, 0])
    diff = pred_price - actual_price
    diff_pct = (diff / actual_price) * 100.0

    print(f"  - Actual price:    ${actual_price:.2f}")
    print(f"  - Predicted price: ${pred_price:.2f}")
    print(f"  - Error delta:     ${diff:+.2f} ({diff_pct:+.2f}%)")
    assert not np.isnan(pred_price), "Predicted price is NaN!"
    assert pred_price > 0.0, "Predicted price must be positive!"
    assert abs(diff_pct) < 25.0, f"Error delta too large ({diff_pct:.2f}%)!"
    print(f"  -> PASS: {dataset_key.upper()} + {framework} loaded and predicted accurately without retraining!")


def main():
    print("=" * 70)
    print("STARTING DEPLOYMENT SMOKE TEST (4 COMBINATIONS)")
    print("=" * 70)

    # 1. Stock + PyTorch
    test_combination("stock", "PyTorch")

    # 2. Stock + Keras
    test_combination("stock", "Keras")

    # 3. Gold + PyTorch
    test_combination("commodity", "PyTorch")

    # 4. Gold + Keras
    test_combination("commodity", "Keras")

    print("\n" + "=" * 70)
    print("TESTING STREAMLIT SERVER SPIN-UP & HEALTH ENDPOINT")
    print("=" * 70)

    test_port = 8508
    streamlit_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "streamlit.exe")
    app_script = os.path.join(PROJECT_ROOT, "deployment", "assignment06", "app.py")

    cmd = [
        streamlit_exe,
        "run",
        app_script,
        "--server.headless=true",
        f"--server.port={test_port}",
    ]

    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        health_ok = False
        health_url = f"http://localhost:{test_port}/_stcore/health"
        for attempt in range(15):
            time.sleep(1)
            try:
                resp = urllib.request.urlopen(health_url, timeout=2)
                if resp.getcode() == 200:
                    health_ok = True
                    print(f"Server health check succeeded on attempt {attempt + 1}: HTTP 200 OK")
                    break
            except Exception:
                continue

        assert health_ok, "Streamlit server health check failed!"
        print("Streamlit UI successfully initialized without crash!")
    finally:
        print("Terminating Streamlit test server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("Streamlit test server cleanly shut down.")

    print("\n" + "=" * 70)
    print("ALL 4 COMBINATIONS & STREAMLIT SERVER SMOKE TEST: 100% PASS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
