"""
Model Evaluation Metrics

Comprehensive evaluation metrics for cryptocurrency price prediction models.
Includes standard regression metrics and domain-specific financial metrics.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate standard regression metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy - how often the model predicts the correct trend direction.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Directional accuracy percentage
    """
    if len(y_true) < 2:
        return 0.0

    # Calculate actual and predicted directions
    actual_direction = np.diff(y_true)
    predicted_direction = np.diff(y_pred)

    # Convert to binary (1 for up, -1 for down, 0 for no change)
    actual_direction = np.sign(actual_direction)
    predicted_direction = np.sign(predicted_direction)

    # Calculate accuracy
    correct_predictions = np.sum(actual_direction == predicted_direction)
    total_predictions = len(actual_direction)

    return (correct_predictions / total_predictions) * 100


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for trading strategy evaluation.

    Args:
        returns: Daily returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    daily_risk_free = risk_free_rate / 252  # Convert annual to daily
    excess_returns = returns - daily_risk_free

    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_max_drawdown(prices: np.ndarray) -> float:
    """
    Calculate maximum drawdown.

    Args:
        prices: Price series

    Returns:
        Maximum drawdown percentage
    """
    if len(prices) == 0:
        return 0.0

    # Calculate running maximum
    peak = np.maximum.accumulate(prices)

    # Calculate drawdown
    drawdown = (prices - peak) / peak

    return np.min(drawdown) * 100


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex = None,
    return_detailed: bool = False,
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Date index for time series analysis
        return_detailed: Whether to return detailed breakdown

    Returns:
        Dictionary of evaluation metrics
    """
    # Basic regression metrics
    metrics = calculate_regression_metrics(y_true, y_pred)

    # Directional accuracy
    metrics["directional_accuracy"] = calculate_directional_accuracy(y_true, y_pred)

    # Financial metrics if we have enough data
    if len(y_true) > 1:
        # Calculate returns
        actual_returns = np.diff(y_true) / y_true[:-1]
        predicted_returns = np.diff(y_pred) / y_pred[:-1]

        # Correlation of returns
        if np.std(actual_returns) > 0 and np.std(predicted_returns) > 0:
            metrics["return_correlation"] = np.corrcoef(
                actual_returns, predicted_returns
            )[0, 1]
        else:
            metrics["return_correlation"] = 0.0

        # Sharpe ratio (for actual returns)
        metrics["sharpe_ratio_actual"] = calculate_sharpe_ratio(actual_returns)

        # Max drawdown
        metrics["max_drawdown_actual"] = calculate_max_drawdown(y_true)
        metrics["max_drawdown_predicted"] = calculate_max_drawdown(y_pred)

    # Add relative performance metrics
    metrics["rmse_normalized"] = metrics["rmse"] / np.mean(y_true)
    metrics["mae_normalized"] = metrics["mae"] / np.mean(y_true)

    return metrics


def compare_models(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple model results in a nice DataFrame.

    Args:
        results_dict: Dictionary with model names as keys and metrics as values

    Returns:
        Comparison DataFrame
    """
    comparison_df = pd.DataFrame(results_dict).T

    # Round for better display
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(4)

    return comparison_df


def compute_prediction_intervals(
    model: torch.nn.Module,
    X: torch.Tensor,
    n_samples: int = 100,
    confidence: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction intervals using Monte Carlo dropout.

    Args:
        model: Trained PyTorch model
        X: Input tensor
        n_samples: Number of MC samples
        confidence: Confidence level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    model.train()  # Enable dropout
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X).cpu().numpy()
            predictions.append(pred)

    predictions = np.array(predictions)

    # Calculate confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions, upper_percentile, axis=0)

    return lower_bound.flatten(), upper_bound.flatten()
