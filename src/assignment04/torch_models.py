"""PyTorch definitions matching the NumPy architectures (NCL layout)."""

import torch
from torch import nn


def _masked_global_max(x, mask=None):
    if mask is None:
        return x.amax(dim=2)
    valid = mask[:, None, :]
    output = x.masked_fill(~valid, torch.finfo(x.dtype).min).amax(dim=2)
    return torch.where(mask.any(dim=1, keepdim=True), output, torch.zeros_like(output))


def _pool_mask(mask):
    usable = (mask.shape[1] // 2) * 2
    return mask[:, :usable].reshape(mask.shape[0], -1, 2).any(dim=2)


class TorchCNN(nn.Module):
    def __init__(self, input_channels, output_dim, *, improved=False):
        super().__init__()
        self.improved = improved
        self.conv1 = nn.Conv1d(input_channels, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        if improved:
            self.conv3 = nn.Conv1d(16, 32, 3, padding=1)
            self.hidden = nn.Linear(32, 16)
            self.output = nn.Linear(16, output_dim)
        else:
            self.output = nn.Linear(16, output_dim)

    def forward(self, x, mask=None):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        pooled_mask = None if mask is None else _pool_mask(mask)
        if self.improved:
            x = torch.relu(self.conv3(x))
        x = _masked_global_max(x, pooled_mask)
        if self.improved:
            x = torch.relu(self.hidden(x))
        return self.output(x)

    @property
    def trainable_layers(self):
        return 5 if self.improved else 3


class TorchTextCNN(nn.Module):
    def __init__(self, vocabulary_size, output_dim=2, *, improved=False):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 16, padding_idx=0)
        self.cnn = TorchCNN(16, output_dim, improved=improved)

    def forward(self, ids, mask=None):
        effective_mask = ids.ne(0) if mask is None else mask
        return self.cnn(self.embedding(ids).transpose(1, 2), effective_mask)


class TorchHouseFieldEncoder(nn.Module):
    def __init__(self, state_size, status_size):
        super().__init__()
        self.numeric_weight = nn.Parameter(torch.empty(4, 8))
        self.numeric_bias = nn.Parameter(torch.zeros(4, 8))
        nn.init.normal_(self.numeric_weight, std=0.1)
        self.state_embedding = nn.Embedding(state_size, 8)
        self.status_embedding = nn.Embedding(status_size, 8)

    def forward(self, numeric, state_ids, status_ids):
        numeric_tokens = numeric[:, :, None] * self.numeric_weight[None, :, :] + self.numeric_bias[None, :, :]
        fields = torch.cat([
            self.state_embedding(state_ids)[:, None, :],
            self.status_embedding(status_ids)[:, None, :],
            numeric_tokens,
        ], dim=1)
        return fields.transpose(1, 2)


class TorchHouseCNN(nn.Module):
    def __init__(self, state_size, status_size, *, improved=False):
        super().__init__()
        self.encoder = TorchHouseFieldEncoder(state_size, status_size)
        self.cnn = TorchCNN(8, 1, improved=improved)

    def forward(self, numeric, state_ids, status_ids):
        return self.cnn(self.encoder(numeric, state_ids, status_ids))
