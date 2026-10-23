"""PyTorch Recurrent Neural Network (Vanilla RNN) implementation.

Strictly adheres to Vanilla RNN (Elman RNN) using torch.nn.RNN.
Does NOT substitute with LSTM or GRU as required by the assignment.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class VanillaRNNPyTorch(nn.Module):
    """Vanilla Recurrent Neural Network (Elman RNN) for Time-Series Regression.

    Architecture:
        Input Sequence: [x_1, x_2, ..., x_T] with shape (batch_size, seq_len, input_size)
        -> nn.RNN (Elman RNN with tanh non-linearity and weight sharing)
        -> Extract last hidden state h_T with shape (batch_size, hidden_size)
        -> nn.Linear(hidden_size, output_size)
        -> Next-step prediction y_hat with shape (batch_size, output_size)

    Formulation:
        h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
        y_hat = W_ho @ h_T + b_ho
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout
        self.nonlinearity = nonlinearity

        # Core recurrent layer
        # PyTorch requires dropout == 0 if num_layers == 1
        effective_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,  # Input shape: (batch, seq, feature)
            dropout=effective_dropout,
        )

        # Fully connected head
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward propagation through time.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h0: Optional initial hidden state of shape (num_layers, batch_size, hidden_size)

        Returns:
            predictions: Tensor of shape (batch_size, output_size)
        """
        # out shape: (batch_size, seq_len, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        out, h_n = self.rnn(x, h0)

        # Many-to-one architecture: take the hidden state corresponding to the final timestep
        # out[:, -1, :] shape: (batch_size, hidden_size)
        last_hidden = out[:, -1, :]

        # Pass through linear projection layer
        # pred shape: (batch_size, output_size)
        pred = self.fc(last_hidden)
        return pred

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, object]:
        """Return model hyperparameter configuration."""
        return {
            "framework": "PyTorch",
            "model_type": "VanillaRNN",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout_rate,
            "nonlinearity": self.nonlinearity,
            "total_parameters": self.count_parameters(),
        }
