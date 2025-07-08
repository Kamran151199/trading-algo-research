"""
Improved CNN-LSTM with Attention for Time Series Forecasting

This improved version addresses the overfitting issues of the original attention model:
- Simplified attention mechanisms
- Better regularization
- Reduced complexity for small datasets
- Improved initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedTemporalAttention(nn.Module):
    """
    Simplified temporal attention mechanism optimized for small datasets
    """

    def __init__(self, hidden_size: int, dropout: float = 0.3):
        super(SimplifiedTemporalAttention, self).__init__()
        self.hidden_size = hidden_size

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in self.attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()

        attention_scores = self.attention(x)
        attention_weights = F.softmax(attention_scores, dim=1)

        attention_weights = self.dropout(attention_weights)

        attended_output = torch.sum(attention_weights * x, dim=1)

        return attended_output, attention_weights.squeeze(-1)


class ImprovedCNNLSTMWithAttention(nn.Module):
    def __init__(
        self,
        num_features: int,
        cnn_filters: int = 64,
        lstm_units: int = 128,
        dropout: float = 0.3,
        use_temporal_attention: bool = True,
    ):
        super(ImprovedCNNLSTMWithAttention, self).__init__()

        self.use_temporal_attention = use_temporal_attention

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=cnn_filters,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_units,
            batch_first=True,
            dropout=dropout if lstm_units > 1 else 0.0,
            num_layers=1,
        )

        if self.use_temporal_attention:
            self.temporal_attention = SimplifiedTemporalAttention(lstm_units, dropout)
            final_input_size = lstm_units
        else:
            final_input_size = lstm_units

        self.fc = nn.Sequential(
            nn.Linear(final_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize all weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x, return_attention_weights=False):
        attention_weights = {}

        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        if self.use_temporal_attention:
            attended_output, temporal_weights = self.temporal_attention(lstm_out)
            attention_weights["temporal_attention"] = temporal_weights
        else:
            attended_output = lstm_out[:, -1, :]

        output = self.fc(attended_output)

        if return_attention_weights:
            return output, attention_weights
        return output


class EnsembleCNNLSTM(nn.Module):
    """
    Ensemble of baseline and improved attention models
    """

    def __init__(self, num_features: int, ensemble_size: int = 3):
        super(EnsembleCNNLSTM, self).__init__()

        self.models = nn.ModuleList()

        for i in range(ensemble_size):
            if i == 0:
                from .cnn_lstm import CNNLSTM

                model = CNNLSTM(num_features=num_features)
            else:
                dropout = 0.2 + i * 0.1
                model = ImprovedCNNLSTMWithAttention(
                    num_features=num_features,
                    dropout=dropout,
                    use_temporal_attention=True,
                )
            self.models.append(model)

    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        return ensemble_pred
