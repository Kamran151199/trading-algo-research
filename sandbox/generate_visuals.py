
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import os

# --- Model Definition ---

class TemporalAttention(nn.Module):
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

# --- Data Loading and Preprocessing ---

def load_and_preprocess_data(file_path, feature_cols, target_col, window_size=30):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    def create_sequences(data, window_size=30):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[feature_cols].iloc[i:i + window_size].values)
            y.append(data[target_col].iloc[i + window_size])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(df_scaled, window_size)
    
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create a scaler for the target column only
    target_scaler = MinMaxScaler()
    target_scaler.fit(df[[target_col]])
    
    return test_loader, target_scaler, df.index[train_size + val_size + window_size:]

# --- Visualization Functions ---

def plot_predictions(actual, predicted, dates, save_path):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Actual Price', color='blue')
    plt.plot(dates, predicted, label='Predicted Price', color='red', linestyle='--')
    plt.title('Bitcoin Price Prediction - Test Set')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(actual, predicted, save_path):
    errors = predicted - actual
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (USD)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_feature_attention(attention_weights, feature_names, save_path):
    avg_attention = np.concatenate([aw['feature_attention'].numpy() for aw in attention_weights], axis=0)
    avg_attention = np.mean(avg_attention, axis=0)
    
    plt.figure(figsize=(12, 8))
    plt.bar(feature_names, avg_attention)
    plt.xticks(rotation=90)
    plt.title('Feature Attention Weights')
    plt.xlabel('Feature')
    plt.ylabel('Attention Weight')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_temporal_attention(attention_weights, save_path):
    avg_attention = np.concatenate([aw['temporal_attention'].numpy() for aw in attention_weights], axis=0)
    avg_attention = np.mean(avg_attention, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_attention)
    plt.title('Temporal Attention Weights')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(actual, predicted):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }

# --- Main Execution ---

if __name__ == '__main__':
    # --- Configuration ---
    DATA_FILE = 'data/processed/btc_with_indicators.csv'
    MODEL_FILE = 'notebooks/bitcoin_models/best_model_checkpoint.pth'
    OUTPUT_DIR = 'figures'
    
    FEATURE_COLS = [
        'open', 'high', 'low', 'close', 'volume',
        'adractcnt', 'txcnt', 'feetotntv',
        'sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_width'
    ]
    TARGET_COL = 'close'
    WINDOW_SIZE = 30
    NUM_FEATURES = len(FEATURE_COLS)

    # --- Create output directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load Model ---
    model = CNNLSTMWithAttention(num_features=NUM_FEATURES)
    checkpoint = torch.load(MODEL_FILE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Load Data ---
    test_loader, target_scaler, test_dates = load_and_preprocess_data(DATA_FILE, FEATURE_COLS, TARGET_COL, WINDOW_SIZE)

    # --- Generate Predictions and Attention Weights ---
    predictions = []
    actuals = []
    attention_weights = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            output, aw = model(batch_X, return_attention_weights=True)
            predictions.extend(output.numpy().flatten())
            actuals.extend(batch_y.numpy().flatten())
            attention_weights.append(aw)

    # --- Inverse Scale ---
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    # --- Generate Visualizations ---
    plot_predictions(actuals, predictions, test_dates, os.path.join(OUTPUT_DIR, 'demo_predictions.png'))
    plot_error_distribution(actuals, predictions, os.path.join(OUTPUT_DIR, 'demo_error_distribution.png'))
    plot_feature_attention(attention_weights, FEATURE_COLS, os.path.join(OUTPUT_DIR, 'demo_feature_attention.png'))
    plot_temporal_attention(attention_weights, os.path.join(OUTPUT_DIR, 'demo_temporal_attention.png'))

    # --- Calculate and Print Metrics ---
    metrics = calculate_metrics(actuals, predictions)
    print("\n--- Performance Metrics ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print(f'\nVisualizations saved to {OUTPUT_DIR}')
