"""TensorFlow / Keras Recurrent Neural Network (Vanilla RNN) implementation.

Strictly adheres to Vanilla RNN (SimpleRNN) using keras.layers.SimpleRNN.
Equivalent architecture and parameterization to VanillaRNNPyTorch.
"""

from typing import Dict, Tuple
import os

# Suppress excessive TensorFlow runtime logging and oneDNN checks
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
from keras import layers, models, optimizers


def build_keras_rnn(
    input_shape: Tuple[int, int] = (30, 5),
    hidden_size: int = 64,
    num_layers: int = 1,
    output_size: int = 1,
    dropout: float = 0.0,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Build an equivalent Vanilla RNN model using Keras SimpleRNN.

    Architecture matches VanillaRNNPyTorch:
        Input: (batch_size, seq_len, input_size)
        -> SimpleRNN layers (tanh non-linearity)
        -> Last hidden state output: (batch_size, hidden_size)
        -> Dense layer: (batch_size, output_size)

    Args:
        input_shape: (seq_len, input_size)
        hidden_size: Dimensionality of RNN hidden state
        num_layers: Number of stacked SimpleRNN layers
        output_size: Dimension of target (1 for next-step price)
        dropout: Recurrent/dense dropout rate
        learning_rate: Initial learning rate for Adam

    Returns:
        Compiled Keras Model
    """
    model = models.Sequential(name="VanillaRNN_Keras")

    # Add input layer explicitly
    model.add(layers.Input(shape=input_shape))

    for layer_idx in range(num_layers):
        is_last_rnn = (layer_idx == num_layers - 1)
        return_sequences = not is_last_rnn

        model.add(
            layers.SimpleRNN(
                units=hidden_size,
                activation="tanh",
                return_sequences=return_sequences,
                dropout=dropout if num_layers > 1 else 0.0,
                name=f"simple_rnn_{layer_idx + 1}",
            )
        )

    # Linear projection head (matching PyTorch nn.Linear)
    model.add(layers.Dense(output_size, activation="linear", name="dense_output"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model


def get_keras_model_config(model: keras.Model) -> Dict[str, object]:
    """Extract configuration dictionary from a Keras model."""
    return {
        "framework": "TensorFlow/Keras",
        "model_type": "SimpleRNN",
        "total_parameters": model.count_params(),
        "layers": [layer.name for layer in model.layers],
    }
