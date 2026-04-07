"""DeepLOB / LOBFrame neural network for limit order book classification.

Reimplementation of the DeepLOB architecture (Zhang, Zohren, Roberts
2019, "DeepLOB: Deep Convolutional Neural Networks for Limit Order
Books") which is the canonical model wrapped by Briola 2024
LOBFrame. Reimplemented from the paper rather than cloning the upstream
repo so it works with vanilla PyTorch and no extra dependencies.

Architecture (DeepLOB v1):
  Input shape: (batch, 1, T_window, 40)

  Block A — convolution over levels (squeeze 40 -> 1)
    Conv2d 1 -> 32  kernel (1, 2)  stride (1, 2)
    LeakyReLU
    Conv2d 32 -> 32 kernel (4, 1)  stride (1, 1) padding (3,0)  -> shrinks T
    LeakyReLU
    Conv2d 32 -> 32 kernel (4, 1)  stride (1, 1) padding (3,0)
    LeakyReLU

  Block B
    Conv2d 32 -> 32 kernel (1, 2)  stride (1, 2)
    LeakyReLU
    Conv2d 32 -> 32 kernel (4, 1)  padding (3,0)
    LeakyReLU
    Conv2d 32 -> 32 kernel (4, 1)  padding (3,0)
    LeakyReLU

  Block C
    Conv2d 32 -> 32 kernel (1, 10)
    LeakyReLU
    Conv2d 32 -> 32 kernel (4, 1)  padding (3,0)
    LeakyReLU
    Conv2d 32 -> 32 kernel (4, 1)  padding (3,0)
    LeakyReLU

  Inception-style merge (3 parallel branches concatenated)

  LSTM 64 hidden

  FC 64 -> n_classes (3 = down/flat/up by default)

Output: logits of shape (batch, n_classes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LeakyConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: tuple[int, int],
                 stride: tuple[int, int] = (1, 1),
                 padding: tuple[int, int] = (0, 0)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                              stride=stride, padding=padding)
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class _Inception(nn.Module):
    """Three parallel branches as in DeepLOB."""

    def __init__(self, in_ch: int = 32, out_ch: int = 64):
        super().__init__()
        # Branch 1: 1x1 -> 3x1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 1)), nn.LeakyReLU(0.01),
            nn.Conv2d(out_ch, out_ch, (3, 1), padding=(1, 0)), nn.LeakyReLU(0.01),
        )
        # Branch 2: 1x1 -> 5x1
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 1)), nn.LeakyReLU(0.01),
            nn.Conv2d(out_ch, out_ch, (5, 1), padding=(2, 0)), nn.LeakyReLU(0.01),
        )
        # Branch 3: maxpool -> 1x1
        self.b3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_ch, out_ch, (1, 1)), nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)


class DeepLOB(nn.Module):
    def __init__(self, n_levels: int = 10, n_classes: int = 3,
                 lstm_hidden: int = 64):
        super().__init__()
        in_features = 4 * n_levels  # 40

        # Block A: squeeze ask/bid pairs
        self.A = nn.Sequential(
            _LeakyConvBlock(1, 32, (1, 2), stride=(1, 2)),
            _LeakyConvBlock(32, 32, (4, 1), padding=(3, 0)),
            _LeakyConvBlock(32, 32, (4, 1), padding=(3, 0)),
        )
        # Block B: squeeze ask|bid further
        self.B = nn.Sequential(
            _LeakyConvBlock(32, 32, (1, 2), stride=(1, 2)),
            _LeakyConvBlock(32, 32, (4, 1), padding=(3, 0)),
            _LeakyConvBlock(32, 32, (4, 1), padding=(3, 0)),
        )
        # Block C: collapse 10 levels to 1 along feature axis
        self.C = nn.Sequential(
            _LeakyConvBlock(32, 32, (1, 10)),
            _LeakyConvBlock(32, 32, (4, 1), padding=(3, 0)),
            _LeakyConvBlock(32, 32, (4, 1), padding=(3, 0)),
        )

        self.inception = _Inception(in_ch=32, out_ch=64)

        self.lstm = nn.LSTM(input_size=192, hidden_size=lstm_hidden,
                            batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T_window, 40)  ->  (batch, 1, T_window, 40)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.A(x)         # (B, 32, T?, 20)
        x = self.B(x)         # (B, 32, T?, 10)
        x = self.C(x)         # (B, 32, T?, 1)
        x = self.inception(x)  # (B, 192, T?, 1)
        # collapse spatial axis to (B, T?, 192)
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)
