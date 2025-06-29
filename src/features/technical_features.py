"""
Technical Indicator Feature Engineering

Adds common trading indicators to market price data using `ta` package.
"""

import pandas as pd
import ta


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard technical indicators to OHLCV DataFrame.

    Args:
        df (pd.DataFrame): Must contain 'open', 'high', 'low', 'close', 'volume'

    Returns:
        pd.DataFrame: Original DataFrame with additional indicator columns
    """
    # Ensure required columns are lowercase
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    # Add indicators
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df["macd"] = ta.trend.macd(df["close"])
    df["macd_signal"] = ta.trend.macd_signal(df["close"])
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    # Clean up NaNs
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    # Quick test on BTC dataset
    filepath = "/Users/komronvalijonov/work/personal/uni-research/data/processed/btc_combined.csv"
    df = pd.read_csv(filepath, parse_dates=["date"])
    df.set_index("date", inplace=True)

    enriched_df = add_technical_indicators(df)
    enriched_df.to_csv("/Users/komronvalijonov/work/personal/uni-research/data/processed/btc_with_indicators.csv")
    print("Saved enriched dataset with indicators.")