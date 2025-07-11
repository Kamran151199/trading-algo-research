"""
CNN-LSTM Hybrid Model for Time Series Forecasting

This model uses 1D convolution to extract local patterns from time-series features,
followed by LSTM to capture temporal dependencies.
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        window_size: int = 30,
        cnn_filters: int = 64,
        lstm_units: int = 128,
    ):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=cnn_filters,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_filters, hidden_size=lstm_units, batch_first=True
        )
        self.fc = nn.Sequential(nn.Linear(lstm_units, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


if __name__ == "__main__":
    model = CNNLSTM(num_features=16, window_size=30)
    dummy_input = torch.randn(8, 30, 16)
    output = model(dummy_input)
    print("Output shape:", output.shape)
