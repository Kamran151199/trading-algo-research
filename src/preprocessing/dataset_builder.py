"""
Dataset Builder

Prepares time-series sequences for deep learning models (LSTM, Transformers).
Includes normalization, sequence creation, and temporal splitting.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_data(
    df: pd.DataFrame, feature_cols: list
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize selected columns using Min-Max scaling.

    Args:
        df: DataFrame with raw features
        feature_cols: List of columns to scale

    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
    return df_scaled, scaler


def create_sequences(
    df: pd.DataFrame, feature_cols: list, target_col: str, window_size: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for supervised learning.

    Args:
        df: Normalized DataFrame
        feature_cols: Input features
        target_col: Prediction target
        window_size: Number of timesteps per sequence

    Returns:
        Tuple of (X, y) NumPy arrays
    """
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[feature_cols].iloc[i : i + window_size].values)
        y.append(df[target_col].iloc[i + window_size])
    return np.array(X), np.array(y)


def temporal_split(
    X: np.ndarray, y: np.ndarray, train_ratio=0.7, val_ratio=0.15
) -> Tuple:
    """
    Split X, y into train, val, and test sets (by time order).

    Returns:
        Tuple of train/val/test splits: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        X[:train_end],
        y[:train_end],
        X[train_end:val_end],
        y[train_end:val_end],
        X[val_end:],
        y[val_end:],
    )


def visualise(X_train, sample_index=0):
    i = sample_index  
    sample = X_train[i]  

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    timesteps = range(sample.shape[0])
    features = range(sample.shape[1])

    for f in features:
        ax.plot(
            timesteps,
            [f] * len(timesteps),
            sample[:, f],
            label=f"Feature {f}",
        )

    ax.set_title("3D View of Sample Window")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Feature Index")
    ax.set_zlabel("Normalized Value")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/komronvalijonov/work/personal/uni-research/data/processed/btc_with_indicators.csv",
        parse_dates=["date"],
    )
    df.set_index("date", inplace=True)

    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adractcnt",
        "txcnt",
        "feetotntv",
        "sma_20",
        "ema_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_upper",
        "bb_lower",
        "bb_width",
    ]
    target = "close"

    df_norm, scaler = normalize_data(df, features)

    X, y = create_sequences(
        df_norm, feature_cols=features, target_col=target, window_size=30
    )

    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y)

    print(f"Data split: {X_train.shape=}, {X_val.shape=}, {X_test.shape=}")
    print(f"Data split: {X_train.shape=}, {X_val.shape=}, {X_test.shape=}")
    print(f"Data split: {X_train.shape=}, {X_val.shape=}, {X_test.shape=}")

    input("Press Enter to visualize a sample...")
    visualise(X_train)
