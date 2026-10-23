# Assignment 06: RNN Deployment Application

Interactive Streamlit web application for time-series price forecasting using Vanilla Recurrent Neural Networks (Elman RNN).

## Overview

This application serves pre-trained Vanilla RNN models built with both **PyTorch** (`nn.RNN`) and **TensorFlow / Keras** (`keras.layers.SimpleRNN`) on two benchmark financial time series:
1. **Stock Market:** Apple Inc. (`AAPL`) Daily Stock Prices
2. **Commodity Market:** Gold Futures (`GC=F`) Continuous Daily Prices

The application performs real-time inference directly from saved model weights (`models/assignment06/`) without requiring model retraining.

## Project Structure

```text
deployment/assignment06/
├── app.py                # Main Streamlit web application
└── README.md             # Deployment instructions & documentation
```

## Running the Application Locally

Ensure the virtual environment is activated:

```powershell
.\.venv\Scripts\Activate.ps1
```

Launch the Streamlit app:

```bash
streamlit run deployment/assignment06/app.py
```

Or run directly with the Python virtual environment:

```bash
.\.venv\Scripts\streamlit.exe run deployment/assignment06/app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Key Features

1. **Dataset Switcher:** Toggle between Apple Stock (AAPL) and Gold Futures (GC=F).
2. **Framework Switcher:** Seamlessly switch between PyTorch (`nn.RNN`) and Keras (`SimpleRNN`).
3. **Interactive Forecast Slider:** Pick any day from the chronological test set (15% split) to evaluate.
4. **Context Window Chart:** Displays the preceding 30 trading days of price history alongside the ground truth and the RNN's predicted next-day price.
5. **Real-time Metrics:** Displays Test MAE, RMSE, $R^2$, and MAPE.
6. **Theoretical Panel:** Explains Elman RNN formulas, unrolled computational graph, weight sharing, BPTT, and gradient clipping.
7. **Cross-Framework Benchmark:** Side-by-side comparison table between PyTorch and Keras.

## Dependencies

- Python 3.12
- `streamlit>=1.30.0`
- `torch>=2.0.0`
- `tensorflow>=2.16.0`
- `keras>=3.0.0`
- `pandas>=2.0.0`
- `numpy>=1.26.0`
- `matplotlib>=3.8.0`
- `scikit-learn>=1.4.0`
