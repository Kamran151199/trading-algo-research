"""
Model Training Module

Comprehensive training pipeline for CNN-LSTM models with proper logging,
checkpointing, and reproducible results.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation.metrics import evaluate_model_predictions


class ModelTrainer:
    """
    Comprehensive training class for CNN-LSTM models
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        save_dir: str = "models",
        experiment_name: str = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.experiment_name = (
            experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self.experiment_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.history = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
            "epochs": [],
            "best_val_loss": float("inf"),
            "best_epoch": 0,
        }

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        batch_size: int = 32,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders for training, validation, and testing
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
        )

        self.test_data = (X_test_t, y_test_t)

        return train_loader, val_loader, test_loader

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 15,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        scheduler_patience: int = 5,
    ) -> Dict:
        """
        Train the model with comprehensive monitoring
        """
        print(f"Starting training experiment: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        print(f"Results will be saved to: {self.experiment_dir}")
        print("-" * 70)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=scheduler_patience, min_lr=1e-7
        )

        best_val_loss = float("inf")
        patience_counter = 0

        start_time = time.time()

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            self.model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    num_val_batches += 1

            val_loss /= num_val_batches

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            self.history["train_losses"].append(train_loss)
            self.history["val_losses"].append(val_loss)
            self.history["learning_rates"].append(current_lr)
            self.history["epochs"].append(epoch + 1)

            if epoch % 5 == 0 or epoch == num_epochs - 1:
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch [{epoch + 1:3d}/{num_epochs}] | "
                    f"Train: {train_loss:.6f} | "
                    f"Val: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed_time:.1f}s"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.history["best_val_loss"] = best_val_loss
                self.history["best_epoch"] = epoch + 1

                self._save_checkpoint(epoch, optimizer, scheduler, train_loss, val_loss)

                if epoch > 10:
                    print(f"New best model! Val Loss: {val_loss:.6f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(
                    f"Best validation loss: {best_val_loss:.6f} at epoch {self.history['best_epoch']}"
                )
                break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")

        self._save_training_history()
        self._plot_training_curves()

        return self.history

    def evaluate(
        self, test_loader: DataLoader, scaler: MinMaxScaler, feature_cols: List[str]
    ) -> Dict:
        """
        Comprehensive model evaluation on test set
        """
        print("\nEvaluating model on test set...")

        self._load_best_model()

        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                pred = self.model(batch_X)
                predictions.extend(pred.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()

        pred_inverse = self._inverse_transform_predictions(
            predictions, scaler, feature_cols
        )
        actual_inverse = self._inverse_transform_predictions(
            actuals, scaler, feature_cols
        )

        metrics = evaluate_model_predictions(actual_inverse, pred_inverse)

        results = {
            "metrics": metrics,
            "predictions_scaled": predictions,
            "actuals_scaled": actuals,
            "predictions_original": pred_inverse,
            "actuals_original": actual_inverse,
        }

        self._save_evaluation_results(results)

        print("Test Results:")
        print(f"   RMSE: ${metrics['rmse']:,.2f}")
        print(f"   MAE: ${metrics['mae']:,.2f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        print(f"   R²: {metrics['r2']:.4f}")
        print(f"   Directional Accuracy: {metrics['directional_accuracy']:.2f}%")

        return results

    def _save_checkpoint(
        self, epoch: int, optimizer, scheduler, train_loss: float, val_loss: float
    ):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": self.history["best_val_loss"],
        }

        checkpoint_path = os.path.join(self.experiment_dir, "best_model_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)

    def _load_best_model(self):
        """Load the best saved model"""
        checkpoint_path = os.path.join(self.experiment_dir, "best_model_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

    def _save_training_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.experiment_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def _plot_training_curves(self):
        """Plot and save training curves"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(
            self.history["epochs"],
            self.history["train_losses"],
            label="Training Loss",
            alpha=0.8,
        )
        plt.plot(
            self.history["epochs"],
            self.history["val_losses"],
            label="Validation Loss",
            alpha=0.8,
        )
        plt.axvline(
            x=self.history["best_epoch"],
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Best Model",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(
            self.history["epochs"],
            self.history["learning_rates"],
            color="orange",
            alpha=0.8,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(self.experiment_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _inverse_transform_predictions(
        self, predictions: np.ndarray, scaler: MinMaxScaler, feature_cols: List[str]
    ) -> np.ndarray:
        """Inverse transform predictions to original scale"""
        dummy_features = np.zeros((len(predictions), len(feature_cols)))
        close_idx = feature_cols.index("close")
        dummy_features[:, close_idx] = predictions

        inversed = scaler.inverse_transform(dummy_features)
        return inversed[:, close_idx]

    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results"""
        metrics_path = os.path.join(self.experiment_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=2)

        predictions_df = pd.DataFrame(
            {
                "actual_scaled": results["actuals_scaled"],
                "predicted_scaled": results["predictions_scaled"],
                "actual_original": results["actuals_original"],
                "predicted_original": results["predictions_original"],
            }
        )

        predictions_path = os.path.join(self.experiment_dir, "test_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
