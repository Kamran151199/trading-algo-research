"""
CNN-LSTM Hybrid Model with Attention for Time Series Forecasting

This enhanced model adds attention mechanisms to the original CNN-LSTM:
- Temporal attention: Focus on important time steps
- Feature attention: Weight different input features
- Both single and multi-head attention variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important time steps
    """
    def __init__(self, hidden_size: int, num_heads: int = 1):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        if num_heads == 1:
            self.attention = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.query = nn.Linear(hidden_size, hidden_size, bias=False)
            self.key = nn.Linear(hidden_size, hidden_size, bias=False)
            self.value = nn.Linear(hidden_size, hidden_size, bias=False)
            self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        
        if self.num_heads == 1:
            attention_scores = self.attention(x)
            attention_weights = F.softmax(attention_scores, dim=1)
            attended_output = torch.sum(attention_weights * x, dim=1)
            return attended_output, attention_weights.squeeze(-1)
        else:
            Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            attended = torch.matmul(attention_weights, V)
            attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            attended = self.out_proj(attended)
            
            attended_output = torch.mean(attended, dim=1)
            
            avg_attention = torch.mean(attention_weights, dim=1)
            avg_attention = torch.mean(avg_attention, dim=-1)
            
            return attended_output, avg_attention


class FeatureAttention(nn.Module):
    """
    Feature attention to weight different input features
    (market data, on-chain metrics, technical indicators)
    """
    def __init__(self, num_features: int):
        super(FeatureAttention, self).__init__()
        self.feature_attention = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        feature_weights = self.feature_attention(torch.mean(x, dim=1))
        feature_weights = feature_weights.unsqueeze(1)
        
        attended_features = x * feature_weights
        return attended_features, feature_weights.squeeze(1)


class CNNLSTMWithAttention(nn.Module):
    def __init__(
        self,
        num_features: int,
        cnn_filters: int = 64,
        lstm_units: int = 128,
        attention_heads: int = 4,
        use_feature_attention: bool = True,
        use_temporal_attention: bool = True,
    ):
        super(CNNLSTMWithAttention, self).__init__()
        
        self.use_feature_attention = use_feature_attention
        self.use_temporal_attention = use_temporal_attention
        
        if self.use_feature_attention:
            self.feature_attention = FeatureAttention(num_features)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=cnn_filters,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters),
            nn.Dropout(0.1),
        )
        
        self.lstm = nn.LSTM(
            input_size=cnn_filters, 
            hidden_size=lstm_units, 
            batch_first=True,
            dropout=0.1
        )
        
        if self.use_temporal_attention:
            self.temporal_attention = TemporalAttention(lstm_units, attention_heads)
            final_input_size = lstm_units
        else:
            final_input_size = lstm_units
        
        self.fc = nn.Sequential(
            nn.Linear(final_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, return_attention_weights=False):
        attention_weights = {}
        
        if self.use_feature_attention:
            x, feature_weights = self.feature_attention(x)
            attention_weights['feature_attention'] = feature_weights
        
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        
        if self.use_temporal_attention:
            attended_output, temporal_weights = self.temporal_attention(lstm_out)
            attention_weights['temporal_attention'] = temporal_weights
        else:
            attended_output = lstm_out[:, -1, :]
        
        output = self.fc(attended_output)
        
        if return_attention_weights:
            return output, attention_weights
        return output
