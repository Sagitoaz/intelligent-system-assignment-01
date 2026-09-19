"""CNN Model Architectures for Assignment 05.

Implements 4 architectures for 2D Image inputs (MNIST, Fashion-MNIST)
and 4 architectures for 1D Tabular inputs (Diabetes BRFSS):
1. Basic CNN (Standard Conv-ReLU-Pool pipeline)
2. LeNet-style CNN (5x5 kernels, subsampling/pooling, multi-stage dense head)
3. VGG-style CNN (Consecutive stacked 3x3 convs, batch normalization, dropout)
4. ResNet-style CNN (Residual blocks with skip connections y = F(x) + x)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# 2D CNN ARCHITECTURES (MNIST & Fashion-MNIST)
# =====================================================================


class BasicCNN2D(nn.Module):
    """Model 1: Basic CNN for 2D images.

    Architecture:
        Input: [B, 1, 28, 28]
        Stage 1: Conv2d(1->16, k=3, p=1) -> ReLU -> MaxPool2d(2, 2)  => [B, 16, 14, 14]
        Stage 2: Conv2d(16->32, k=3, p=1) -> ReLU -> MaxPool2d(2, 2) => [B, 32, 7, 7]
        Head: Flatten -> Linear(32*7*7=1568 -> 64) -> ReLU -> Linear(64 -> num_classes)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    def forward_stages(self, x: torch.Tensor) -> List[Tuple[str, Tuple[int, ...]]]:
        """Trace tensor shapes through intermediate stages."""
        shapes = [("Input", tuple(x.shape))]
        x = self.pool1(self.relu1(self.conv1(x)))
        shapes.append(("Stage 1 (Conv-ReLU-Pool)", tuple(x.shape)))
        x = self.pool2(self.relu2(self.conv2(x)))
        shapes.append(("Stage 2 (Conv-ReLU-Pool)", tuple(x.shape)))
        x = self.flatten(x)
        shapes.append(("Flatten", tuple(x.shape)))
        x = self.relu3(self.fc1(x))
        shapes.append(("Dense Hidden", tuple(x.shape)))
        x = self.fc2(x)
        shapes.append(("Output Logits", tuple(x.shape)))
        return shapes


class LeNetCNN2D(nn.Module):
    """Model 2: LeNet-style CNN for 2D images.

    Architecture (adapted from Yann LeCun's LeNet-5):
        Input: [B, 1, 28, 28]
        Stage 1: Conv2d(1->6, k=5, p=2) -> ReLU -> AvgPool2d(2, 2)   => [B, 6, 14, 14]
        Stage 2: Conv2d(6->16, k=5, p=0) -> ReLU -> AvgPool2d(2, 2)  => [B, 16, 5, 5]
        Head: Flatten -> Linear(16*5*5=400 -> 120) -> ReLU
              -> Linear(120 -> 84) -> ReLU
              -> Linear(84 -> num_classes)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_stages(self, x: torch.Tensor) -> List[Tuple[str, Tuple[int, ...]]]:
        shapes = [("Input", tuple(x.shape))]
        x = self.pool1(self.relu1(self.conv1(x)))
        shapes.append(("LeNet Conv1 (5x5) + AvgPool", tuple(x.shape)))
        x = self.pool2(self.relu2(self.conv2(x)))
        shapes.append(("LeNet Conv2 (5x5) + AvgPool", tuple(x.shape)))
        x = self.flatten(x)
        shapes.append(("Flatten", tuple(x.shape)))
        x = self.relu3(self.fc1(x))
        shapes.append(("Dense FC1 (120)", tuple(x.shape)))
        x = self.relu4(self.fc2(x))
        shapes.append(("Dense FC2 (84)", tuple(x.shape)))
        x = self.fc3(x)
        shapes.append(("Output Logits", tuple(x.shape)))
        return shapes


class VGGCNN2D(nn.Module):
    """Model 3: VGG-style CNN for 2D images.

    VGG Philosophy:
        Replacing large filters with consecutive stacked 3x3 convolutions.
        Two 3x3 convs yield an effective 5x5 receptive field with fewer weights
        and two non-linear activations.

    Architecture:
        Input: [B, 1, 28, 28]
        Block 1:
          Conv2d(1->16, k=3, p=1) -> BN -> ReLU
          Conv2d(16->16, k=3, p=1) -> BN -> ReLU
          MaxPool2d(2, 2)                                            => [B, 16, 14, 14]
        Block 2:
          Conv2d(16->32, k=3, p=1) -> BN -> ReLU
          Conv2d(32->32, k=3, p=1) -> BN -> ReLU
          MaxPool2d(2, 2)                                            => [B, 32, 7, 7]
        Classifier Head:
          Flatten -> Linear(32*7*7=1568 -> 128) -> ReLU -> Dropout(0.4) -> Linear(128 -> num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.4,
    ):
        super().__init__()
        # Block 1: 2 consecutive 3x3 convs
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 2 consecutive 3x3 convs
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

    def forward_stages(self, x: torch.Tensor) -> List[Tuple[str, Tuple[int, ...]]]:
        shapes = [("Input", tuple(x.shape))]
        x = self.block1(x)
        shapes.append(("VGG Block 1 (2x Conv 3x3 + Pool)", tuple(x.shape)))
        x = self.block2(x)
        shapes.append(("VGG Block 2 (2x Conv 3x3 + Pool)", tuple(x.shape)))
        x = self.classifier[0](x)  # Flatten
        shapes.append(("Flatten", tuple(x.shape)))
        x = self.classifier[1](x)  # Linear
        shapes.append(("Dense 128", tuple(x.shape)))
        x = self.classifier[2](x)  # ReLU
        x = self.classifier[3](x)  # Dropout
        x = self.classifier[4](x)  # Output
        shapes.append(("Output Logits", tuple(x.shape)))
        return shapes


class ResidualBlock2D(nn.Module):
    """Residual Block with skip connection: y = F(x) + shortcut(x)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ResNetCNN2D(nn.Module):
    """Model 4: ResNet-style CNN for 2D images.

    ResNet Philosophy:
        Residual connections y = F(x) + x create gradient highways that
        prevent vanishing gradients during backpropagation.
        Replaces large dense flattening layer with Global Average Pooling.

    Architecture:
        Input: [B, 1, 28, 28]
        Initial: Conv2d(1->16, k=3, p=1) -> BN -> ReLU               => [B, 16, 28, 28]
        ResBlock 1: 16 -> 16, stride=1                                => [B, 16, 28, 28]
        ResBlock 2: 16 -> 32, stride=2                                => [B, 32, 14, 14]
        ResBlock 3: 32 -> 64, stride=2                                => [B, 64, 7, 7]
        Head: AdaptiveAvgPool2d((1, 1)) -> Flatten [B, 64] -> Linear(64 -> num_classes)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = ResidualBlock2D(16, 16, stride=1)
        self.layer2 = ResidualBlock2D(16, 32, stride=2)
        self.layer3 = ResidualBlock2D(32, 64, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def forward_stages(self, x: torch.Tensor) -> List[Tuple[str, Tuple[int, ...]]]:
        shapes = [("Input", tuple(x.shape))]
        x = self.initial(x)
        shapes.append(("Initial Conv+BN", tuple(x.shape)))
        x = self.layer1(x)
        shapes.append(("ResBlock 1 (16 ch, stride 1)", tuple(x.shape)))
        x = self.layer2(x)
        shapes.append(("ResBlock 2 (32 ch, stride 2)", tuple(x.shape)))
        x = self.layer3(x)
        shapes.append(("ResBlock 3 (64 ch, stride 2)", tuple(x.shape)))
        x = self.gap(x)
        shapes.append(("Global Avg Pool", tuple(x.shape)))
        x = self.flatten(x)
        shapes.append(("Flatten", tuple(x.shape)))
        x = self.fc(x)
        shapes.append(("Output Logits", tuple(x.shape)))
        return shapes


# =====================================================================
# 1D CNN ARCHITECTURES (Diabetes Tabular)
# =====================================================================


class BasicCNN1D(nn.Module):
    """Model 1 (1D): Basic CNN for Tabular Data.

    Input: [B, 1, 21] (Channel=1, Length=21)
    Stage 1: Conv1d(1->16, k=3, p=1) -> ReLU -> MaxPool1d(2) => [B, 16, 10]
    Stage 2: Conv1d(16->32, k=3, p=1) -> ReLU -> MaxPool1d(2) => [B, 32, 5]
    Head: Flatten -> Linear(32*5=160 -> 32) -> ReLU -> Linear(32 -> 2)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 5, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward_stages(self, x: torch.Tensor) -> List[Tuple[str, Tuple[int, ...]]]:
        shapes = [("Input 1D", tuple(x.shape))]
        x = self.pool1(self.relu1(self.conv1(x)))
        shapes.append(("Stage 1 (Conv1D-ReLU-Pool)", tuple(x.shape)))
        x = self.pool2(self.relu2(self.conv2(x)))
        shapes.append(("Stage 2 (Conv1D-ReLU-Pool)", tuple(x.shape)))
        x = self.flatten(x)
        shapes.append(("Flatten", tuple(x.shape)))
        x = self.relu3(self.fc1(x))
        shapes.append(("Dense 32", tuple(x.shape)))
        x = self.fc2(x)
        shapes.append(("Output Logits", tuple(x.shape)))
        return shapes


class LeNetCNN1D(nn.Module):
    """Model 2 (1D): LeNet-style CNN for Tabular Data.

    Input: [B, 1, 21]
    Stage 1: Conv1d(1->8, k=5, p=2) -> ReLU -> AvgPool1d(2)  => [B, 8, 10]
    Stage 2: Conv1d(8->16, k=5, p=1) -> ReLU -> AvgPool1d(2) => [B, 16, 4]
    Head: Flatten -> Linear(16*4=64 -> 48) -> ReLU -> Linear(48 -> 24) -> ReLU -> Linear(24 -> 2)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4, 48)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(48, 24)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(24, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_stages(self, x: torch.Tensor) -> List[Tuple[str, Tuple[int, ...]]]:
        shapes = [("Input 1D", tuple(x.shape))]
        x = self.pool1(self.relu1(self.conv1(x)))
        shapes.append(("LeNet 1D Stage 1 (Conv 5 + AvgPool)", tuple(x.shape)))
        x = self.pool2(self.relu2(self.conv2(x)))
        shapes.append(("LeNet 1D Stage 2 (Conv 5 + AvgPool)", tuple(x.shape)))
        x = self.flatten(x)
        shapes.append(("Flatten", tuple(x.shape)))
        x = self.relu3(self.fc1(x))
        shapes.append(("Dense 48", tuple(x.shape)))
        x = self.relu4(self.fc2(x))
        shapes.append(("Dense 24", tuple(x.shape)))
        x = self.fc3(x)
        shapes.append(("Output Logits", tuple(x.shape)))
        return shapes


class VGGCNN1D(nn.Module):
    """Model 3 (1D): VGG-style CNN for Tabular Data.

    Block 1: Conv1d(1->16, 3, p=1) -> BN -> ReLU -> Conv1d(16->16, 3, p=1) -> BN -> ReLU -> MaxPool1d(2) => [B, 16, 10]
    Block 2: Conv1d(16->32, 3, p=1) -> BN -> ReLU -> Conv1d(32->32, 3, p=1) -> BN -> ReLU -> MaxPool1d(2) => [B, 32, 5]
    Head: Flatten -> Linear(32*5=160 -> 64) -> ReLU -> Dropout(0.3) -> Linear(64 -> 2)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

    def forward_stages(self, x: torch.Tensor) -> List[Tuple[str, Tuple[int, ...]]]:
        shapes = [("Input 1D", tuple(x.shape))]
        x = self.block1(x)
        shapes.append(("VGG 1D Block 1 (2x Conv 3 + Pool)", tuple(x.shape)))
        x = self.block2(x)
        shapes.append(("VGG 1D Block 2 (2x Conv 3 + Pool)", tuple(x.shape)))
        x = self.classifier(x)
        shapes.append(("Output Logits", tuple(x.shape)))
        return shapes


class ResidualBlock1D(nn.Module):
    """1D Residual Block with skip connection: y = F(x) + shortcut(x)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out


class ResNetCNN1D(nn.Module):
    """Model 4 (1D): ResNet-style CNN for Tabular Data.

    Input: [B, 1, 21]
    Initial: Conv1d(1->16, 3, p=1) -> BN -> ReLU                     => [B, 16, 21]
    ResBlock 1: 16 -> 16, stride=1                                    => [B, 16, 21]
    ResBlock 2: 16 -> 32, stride=2                                    => [B, 32, 11]
    Head: AdaptiveAvgPool1d(1) -> Flatten [B, 32] -> Linear(32 -> 2)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1 = ResidualBlock1D(16, 16, stride=1)
        self.layer2 = ResidualBlock1D(16, 32, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def forward_stages(self, x: torch.Tensor) -> List[Tuple[str, Tuple[int, ...]]]:
        shapes = [("Input 1D", tuple(x.shape))]
        x = self.initial(x)
        shapes.append(("Initial Conv1D + BN", tuple(x.shape)))
        x = self.layer1(x)
        shapes.append(("ResBlock 1D-1 (16 ch)", tuple(x.shape)))
        x = self.layer2(x)
        shapes.append(("ResBlock 1D-2 (32 ch, stride 2)", tuple(x.shape)))
        x = self.gap(x)
        shapes.append(("Global Avg Pool 1D", tuple(x.shape)))
        x = self.flatten(x)
        shapes.append(("Flatten", tuple(x.shape)))
        x = self.fc(x)
        shapes.append(("Output Logits", tuple(x.shape)))
        return shapes
