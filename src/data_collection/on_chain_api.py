"""
CoinMetrics On-Chain Data API Interface

This module fetches on-chain metrics using the official CoinMetrics Python SDK.
"""

from datetime import datetime

import pandas as pd
from coinmetrics.api_client import CoinMetricsClient

# Initialize the client (no API key required for Community API)
client = CoinMetricsClient()


def fetch_onchain_metrics(
    asset: str, metrics: list, start: str, end: str
) -> pd.DataFrame:
    """
    Fetch on-chain metrics for a given cryptocurrency asset.

    Args:
        asset (str): Asset symbol (e.g., 'btc')
        metrics (list): List of metric identifiers to fetch
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: Time series DataFrame with selected metrics
    """
    df = client.get_asset_metrics(
        assets=[asset], metrics=metrics, frequency="1d", start_time=start, end_time=end
    ).to_dataframe()

    df["asset"] = asset
    df["download_timestamp"] = datetime.now()
    return df


if __name__ == "__main__":
    # Sample test for BTC on-chain metrics
    asset = "btc"
    metrics = ["AdrActCnt", "TxCnt", "FeeTotNtv"]
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    df = fetch_onchain_metrics(asset, metrics, start_date, end_date)
    print(df.head())

    print(f"Fetched {len(df)} records for {asset} from {start_date} to {end_date}")

    # Save to CSV for further processing
    output_path = "/Users/komronvalijonov/work/personal/uni-research/data/raw/btc_onchain_metrics.csv"
    df.to_csv(output_path, index=False)
