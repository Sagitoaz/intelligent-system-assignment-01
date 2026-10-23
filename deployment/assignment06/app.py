"""Streamlit Web Application for Assignment 06: RNN Time-Series Forecasting.

Features:
- Select Dataset: Stock (Apple - AAPL) or Commodity (Gold - GC=F)
- Select Framework: PyTorch (nn.RNN) or TensorFlow/Keras (SimpleRNN)
- Loads pre-trained model weights from models/assignment06/ (Zero retraining on launch)
- Interactive Historical vs Test Period Chart
- Window Selection / Next-day Price Prediction
- Ground Truth Context, Prediction Error, and Percentage Delta
- Model Architecture & Theoretical Mechanics Panel
"""

import json
import os
import sys

# Suppress TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure repository root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.assignment06.data import (
    prepare_time_series_data,
    get_dataset_metadata,
    TimeSeriesDataBundle,
)
from src.assignment06.models_pytorch import VanillaRNNPyTorch
from src.assignment06.evaluation import calculate_regression_metrics

# Streamlit Page Configuration
st.set_page_config(
    page_title="RNN Time-Series Forecasting Demo | A06",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_bundle(dataset_name: str) -> TimeSeriesDataBundle:
    """Load and cache preprocessed time-series bundle."""
    return prepare_time_series_data(dataset_name, seq_len=30)


@st.cache_resource(show_spinner=False)
def load_pytorch_model(dataset_name: str, input_size: int, hidden_size: int = 64) -> VanillaRNNPyTorch:
    """Load pre-trained PyTorch Vanilla RNN weights."""
    model_path = os.path.join(PROJECT_ROOT, "models", "assignment06", f"{dataset_name}_rnn_pytorch.pt")
    model = VanillaRNNPyTorch(input_size=input_size, hidden_size=hidden_size, num_layers=1, output_size=1)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_keras_model(dataset_name: str):
    """Load pre-trained Keras Vanilla RNN model."""
    import keras
    model_path = os.path.join(PROJECT_ROOT, "models", "assignment06", f"{dataset_name}_rnn_keras.keras")
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    return None


@st.cache_data(show_spinner=False)
def load_results_summary(dataset_name: str):
    """Load evaluation metrics JSON file."""
    json_path = os.path.join(PROJECT_ROOT, "results", "assignment06", f"{dataset_name}_metrics.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None


# Sidebar Navigation & Selection
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
st.sidebar.title("Configuration")
st.sidebar.markdown("**Assignment 06: Vanilla RNN**")

dataset_choice = st.sidebar.selectbox(
    "Choose Dataset:",
    options=["Stock (Apple Inc. - AAPL)", "Commodity (Gold Futures - GC=F)"],
    index=0,
)
dataset_key = "stock" if "Stock" in dataset_choice else "commodity"
meta = get_dataset_metadata(dataset_key)

framework_choice = st.sidebar.radio(
    "Select Model Framework:",
    options=["PyTorch (nn.RNN)", "TensorFlow / Keras (SimpleRNN)"],
    index=0,
)
is_pytorch = "PyTorch" in framework_choice

st.sidebar.divider()
st.sidebar.markdown(f"**Asset:** {meta['name']}")
st.sidebar.markdown(f"**Target:** {meta['primary_target']}")
st.sidebar.markdown(f"**Sequence Window:** 30 Days")
st.sidebar.markdown(f"**Features (5):** Open, High, Low, Close, Volume")

# Main Page Header
st.title("📈 Vanilla RNN Time-Series Forecasting")
st.markdown(
    f"Predicting next-day closing price for **{meta['name']}** using "
    f"a standard Elman Recurrent Neural Network in **{framework_choice}**."
)

# Load Data
with st.spinner("Loading preprocessed dataset..."):
    bundle = get_cached_bundle(dataset_key)
    metrics_summary = load_results_summary(dataset_key)

# Overview KPIs
y_test_orig = bundle.target_scaler.inverse_transform(bundle.y_test).ravel()
test_dates = pd.to_datetime(bundle.test_dates)

col1, col2, col3, col4 = st.columns(4)

framework_metrics = None
if metrics_summary:
    fw_key = "pytorch" if is_pytorch else "keras"
    framework_metrics = metrics_summary.get(fw_key, {}).get("metrics", {})

mae_val = framework_metrics.get("MAE", 0.0) if framework_metrics else 0.0
rmse_val = framework_metrics.get("RMSE", 0.0) if framework_metrics else 0.0
r2_val = framework_metrics.get("R2", 0.0) if framework_metrics else 0.0
mape_val = framework_metrics.get("MAPE", 0.0) if framework_metrics else 0.0

with col1:
    st.metric("Test MAE", f"${mae_val:.2f}" if mae_val > 0 else "N/A")
with col2:
    st.metric("Test RMSE", f"${rmse_val:.2f}" if rmse_val > 0 else "N/A")
with col3:
    st.metric("Test R² Score", f"{r2_val:.4f}" if r2_val != 0.0 else "N/A")
with col4:
    st.metric("Test MAPE", f"{mape_val:.2f}%" if mape_val > 0 else "N/A")

st.divider()

# Tab Layout
tab_predict, tab_test_chart, tab_architecture, tab_comparison = st.tabs(
    ["🎯 Next-Day Prediction", "📊 Test Set Trajectory", "🧠 RNN Architecture & Math", "⚖️ PyTorch vs. Keras"]
)

# -------------------------------------------------------------
# TAB 1: Interactive Prediction
# -------------------------------------------------------------
with tab_predict:
    st.subheader("Select a Target Date in Test Split to Forecast")
    
    num_test_samples = len(bundle.X_test)
    sample_idx = st.slider(
        "Select Time Window Index:",
        min_value=0,
        max_value=num_test_samples - 1,
        value=num_test_samples - 1,
        help="Choose a day from the test set. The model uses the preceding 30 days of data to predict that day's closing price.",
    )

    chosen_date = test_dates.iloc[sample_idx].strftime("%Y-%m-%d")
    actual_price = float(y_test_orig[sample_idx])

    # Model inference
    window_x = bundle.X_test[sample_idx : sample_idx + 1]  # shape (1, 30, 5)

    if is_pytorch:
        pt_model = load_pytorch_model(dataset_key, input_size=5, hidden_size=64)
        with torch.no_grad():
            tensor_x = torch.from_numpy(window_x).float()
            pred_scaled = pt_model(tensor_x).cpu().numpy()
    else:
        kr_model = load_keras_model(dataset_key)
        if kr_model is not None:
            pred_scaled = kr_model.predict(window_x, verbose=0)
        else:
            st.error("Keras model checkpoint not found!")
            pred_scaled = np.zeros((1, 1))

    pred_price = float(bundle.target_scaler.inverse_transform(pred_scaled).ravel()[0])
    diff_val = pred_price - actual_price
    diff_pct = (diff_val / actual_price) * 100

    # Display prediction results
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.markdown(f"**Target Forecast Date:**\n### 📅 {chosen_date}")
    with pcol2:
        st.markdown(f"**Actual Close Price:**\n### 💵 ${actual_price:.2f}")
    with pcol3:
        st.markdown(f"**Model Predicted Close:**\n### 🔮 ${pred_price:.2f}")
    with pcol4:
        delta_color = "normal" if abs(diff_pct) < 2.0 else "inverse"
        st.metric(
            label="Prediction Delta (Error)",
            value=f"${diff_val:+.2f}",
            delta=f"{diff_pct:+.2f}%",
            delta_color=delta_color,
        )

    # Context window chart
    st.markdown("#### Input Window (Past 30 Days) & Next-Step Forecast")
    
    # Retrieve past 30 days of unscaled close prices
    # Note: features scaler inverse
    unscaled_window_feat = bundle.feature_scaler.inverse_transform(window_x[0])
    close_idx = bundle.feature_cols.index("Close")
    past_30_close = unscaled_window_feat[:, close_idx]

    fig_ctx, ax_ctx = plt.subplots(figsize=(12, 4.5))
    x_steps = list(range(-30, 0))
    ax_ctx.plot(x_steps, past_30_close, marker="o", markersize=4, color="#3b82f6", lw=2, label="Historical Window (t-29 to t)")
    ax_ctx.scatter([1], [actual_price], color="#0f172a", s=100, zorder=5, label=f"Actual Price ({chosen_date}): ${actual_price:.2f}")
    ax_ctx.scatter([1], [pred_price], color="#dc2626", marker="^", s=130, zorder=6, label=f"Predicted Price ({chosen_date}): ${pred_price:.2f}")
    ax_ctx.plot([0, 1], [past_30_close[-1], actual_price], color="#64748b", linestyle=":", lw=1.5)
    ax_ctx.plot([0, 1], [past_30_close[-1], pred_price], color="#dc2626", linestyle="--", lw=1.5)

    ax_ctx.set_xlabel("Relative Timestep (Days relative to forecast)", fontsize=11)
    ax_ctx.set_ylabel("Close Price (USD)", fontsize=11)
    ax_ctx.set_title(f"30-Day Context Window -> Timestep t+1 Forecast ({meta['name']})", fontsize=12, fontweight="bold")
    ax_ctx.grid(True, linestyle=":", alpha=0.6)
    ax_ctx.legend(loc="upper left", frameon=True)
    st.pyplot(fig_ctx)
    plt.close(fig_ctx)

# -------------------------------------------------------------
# TAB 2: Test Set Trajectory
# -------------------------------------------------------------
with tab_test_chart:
    st.subheader("Test Split Performance (Ground Truth vs. Predictions)")
    pred_fig_path = os.path.join(PROJECT_ROOT, "figures", "assignment06", f"{dataset_key}_actual_vs_predicted.png")
    if os.path.exists(pred_fig_path):
        st.image(pred_fig_path, caption=f"Test Set Prediction Trajectory ({meta['name']})", use_container_width=True)
    else:
        st.info("Test trajectory figure will display once training is complete.")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        res_fig_path = os.path.join(PROJECT_ROOT, "figures", "assignment06", f"{dataset_key}_residuals.png")
        if os.path.exists(res_fig_path):
            st.image(res_fig_path, caption="Residual Analysis (Error Distribution)", use_container_width=True)
    with col_res2:
        loss_fig_path = os.path.join(PROJECT_ROOT, "figures", "assignment06", f"{dataset_key}_loss_curves.png")
        if os.path.exists(loss_fig_path):
            st.image(loss_fig_path, caption="Training & Validation Loss Curves", use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Theoretical Mechanics
# -------------------------------------------------------------
with tab_architecture:
    st.subheader("Recurrent Neural Network Mathematical Formulation")
    st.markdown(
        r"""
        ### 1. Elman RNN Cell Equations
        At each discrete timestep $t \in \{1, 2, \dots, T\}$:
        
        $$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$
        
        Where:
        - $x_t \in \mathbb{R}^{d_{in}}$: Input feature vector at timestep $t$ ($d_{in} = 5$ features: Open, High, Low, Close, Volume).
        - $h_t \in \mathbb{R}^{d_h}$: Hidden state vector capturing temporal memory up to timestep $t$ ($d_h = 64$).
        - $W_{ih} \in \mathbb{R}^{d_h \times d_{in}}$: Input-to-hidden weight matrix (**shared across all timesteps**).
        - $W_{hh} \in \mathbb{R}^{d_h \times d_h}$: Hidden-to-hidden recurrent transition matrix (**shared across all timesteps**).
        - $b_{ih}, b_{hh} \in \mathbb{R}^{d_h}$: Bias vectors.

        ### 2. Many-to-One Readout
        After processing the final timestep $T = 30$, the terminal hidden state $h_T$ summarizes the sequence:
        
        $$\hat{y}_{t+1} = W_{ho} h_T + b_o$$
        
        Where $W_{ho} \in \mathbb{R}^{1 \times d_h}$ and $\hat{y}_{t+1}$ is the predicted next-day closing price.

        ### 3. Backpropagation Through Time (BPTT) & Gradient Challenges
        During training, the error gradient is backpropagated backward through all $T$ timesteps:
        
        $$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial h_T} \left( \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}} \right) \frac{\partial h_t}{\partial W_{hh}}$$
        
        Because $\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(1 - h_k^2) W_{hh}^T$, chaining $T$ matrix multiplications causes:
        - **Vanishing Gradients**: If eigenvalues of $W_{hh} < 1$, gradients exponentially shrink towards 0 as $T \to \infty$.
        - **Exploding Gradients**: If eigenvalues $> 1$, gradients exponentially explode. We mitigate this using **gradient norm clipping** (`clip_grad_norm_ = 1.0`).
        """
    )

# -------------------------------------------------------------
# TAB 4: Framework Comparison
# -------------------------------------------------------------
with tab_comparison:
    st.subheader("PyTorch (nn.RNN) vs. TensorFlow/Keras (SimpleRNN)")
    csv_path = os.path.join(PROJECT_ROOT, "results", "assignment06", "overall_comparison.csv")
    if os.path.exists(csv_path):
        df_comp = pd.read_csv(csv_path)
        st.dataframe(df_comp, use_container_width=True)
    else:
        st.info("Overall comparison table will appear once all models finish training.")

    st.markdown(
        """
        ### Observations & Framework Parity:
        1. **Architectural Parity:** Both implementations share identical 1-layer Vanilla RNN architectures with 64 hidden units, tanh activation, and a single linear dense readout layer.
        2. **Convergence Speed:** PyTorch executes with manual explicit loops in pure Python with torch tensor operations, while Keras wraps TensorFlow's compiled graph operations.
        3. **Stability:** Both frameworks reach competitive regression metrics (low MAE and RMSE) when trained with Adam on sliding windows of length 30.
        """
    )

st.divider()
st.caption("Intelligent System Development | Assignment 06: Recurrent Neural Network (Vanilla RNN)")
