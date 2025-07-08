#!/usr/bin/env python3

import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.append("src")

from src.models.cnn_lstm import CNNLSTM
from src.models.improved_cnn_lstm_attention import ImprovedCNNLSTMWithAttention
from src.preprocessing.dataset_builder import create_sequences
from src.training.trainer import ModelTrainer

warnings.filterwarnings("ignore")


def main():
    print("Cryptocurrency Price Prediction - Project Demonstration")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading Bitcoin price data...")

    try:
        df = pd.read_csv("data/processed/btc_with_indicators.csv", parse_dates=["date"])
        df.set_index("date", inplace=True)
        print(f"Data loaded: {df.shape[0]} days, {df.shape[1]} features")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
    except FileNotFoundError:
        print(
            "Data file not found. Please ensure data/processed/btc_with_indicators.csv exists"
        )
        return

    feature_cols = [
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
    target_col = "close"

    print(f"Features: {len(feature_cols)} features")
    print(f"Target: {target_col}")

    print("\nPreprocessing data...")
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    window_size = 30
    X, y = create_sequences(df_scaled, feature_cols, target_col, window_size)

    print(f"Sequences created: X{X.shape}, y{y.shape}")

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = (
        X[train_size : train_size + val_size],
        y[train_size : train_size + val_size],
    )
    X_test, y_test = X[train_size + val_size :], y[train_size + val_size :]

    print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n" + "=" * 50)
    print("DEMONSTRATION 1: Baseline CNN-LSTM Model")
    print("=" * 50)

    print("Creating baseline CNN-LSTM model...")
    baseline_model = CNNLSTM(num_features=len(feature_cols))
    print(f"Model parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")

    trainer_baseline = ModelTrainer(
        model=baseline_model,
        device="cpu",
        save_dir="demo_models",
        experiment_name="baseline_demo",
    )

    train_loader, val_loader, test_loader = trainer_baseline.prepare_data(
        X, y, batch_size=32
    )

    print("Training baseline model (quick demo - 10 epochs)...")
    history_baseline = trainer_baseline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=0.001,
        patience=5,
    )

    results_baseline = trainer_baseline.evaluate(test_loader, scaler, feature_cols)
    print("Baseline Results:")
    print(f"   RMSE: ${results_baseline['metrics']['rmse']:,.2f}")
    print(f"   MAPE: {results_baseline['metrics']['mape']:.2f}%")
    print(
        f"   Directional Accuracy: {results_baseline['metrics']['directional_accuracy']:.2f}%"
    )

    print("\n" + "=" * 50)
    print("DEMONSTRATION 2: Improved Attention Model")
    print("=" * 50)

    print("Creating improved attention model...")
    attention_model = ImprovedCNNLSTMWithAttention(
        num_features=len(feature_cols), dropout=0.4, use_temporal_attention=True
    )
    print(f"Model parameters: {sum(p.numel() for p in attention_model.parameters()):,}")

    trainer_attention = ModelTrainer(
        model=attention_model,
        device="cpu",
        save_dir="demo_models",
        experiment_name="attention_demo",
    )

    train_loader, val_loader, test_loader = trainer_attention.prepare_data(
        X, y, batch_size=32
    )

    print("Training attention model (quick demo - 10 epochs)...")
    history_attention = trainer_attention.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=0.0005,
        patience=5,
    )

    results_attention = trainer_attention.evaluate(test_loader, scaler, feature_cols)
    print("Attention Results:")
    print(f"   RMSE: ${results_attention['metrics']['rmse']:,.2f}")
    print(f"   MAPE: {results_attention['metrics']['mape']:.2f}%")
    print(
        f"   Directional Accuracy: {results_attention['metrics']['directional_accuracy']:.2f}%"
    )

    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)

    baseline_rmse = results_baseline["metrics"]["rmse"]
    attention_rmse = results_attention["metrics"]["rmse"]

    improvement = ((baseline_rmse - attention_rmse) / baseline_rmse) * 100

    print(f"Baseline CNN-LSTM:      RMSE ${baseline_rmse:,.2f}")
    print(f"Improved Attention:     RMSE ${attention_rmse:,.2f}")
    print(f"Improvement:            {improvement:+.1f}%")

    if improvement > 0:
        print("SUCCESS: Attention model improved over baseline!")
    else:
        print("Baseline still better (this is expected with quick demo training)")

    print("\nSample Predictions:")
    if "predictions" in results_attention:
        sample_preds = results_attention["predictions"][:5]
        sample_actual = results_attention["actual_values"][:5]

        print("Date       | Actual    | Predicted | Error")
        print("-" * 45)
        for i, (actual, pred) in enumerate(zip(sample_actual, sample_preds)):
            error = pred - actual
            print(
                f"Day {i + 1:2d}     | ${actual:8,.0f} | ${pred:9,.0f} | ${error:+7,.0f}"
            )

    print("\nDemonstration completed!")
    print("For complete results, see:")
    print("   - research_report.md: Full research documentation")
    print("   - notebooks/: Interactive Jupyter demonstrations")
    print("   - figures/: Generated visualizations")

    import shutil

    try:
        shutil.rmtree("demo_models")
        print("Demo models cleaned up")
    except:
        pass


if __name__ == "__main__":
    main()
