"""
Data Preprocessing Pipeline

Combines and cleans market and on-chain data sources for modeling.
"""

import pandas as pd


def align_and_merge_data(
    market_df: pd.DataFrame, onchain_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Align market and on-chain data on date index.

    Args:
        market_df (pd.DataFrame): DataFrame from Alpha Vantage
        onchain_df (pd.DataFrame): DataFrame from CoinMetrics

    Returns:
        pd.DataFrame: Unified DataFrame with aligned and merged data
    """
    onchain_df.rename(columns={"time": "date"}, inplace=True)
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.tz_localize(None)
    onchain_df["date"] = pd.to_datetime(onchain_df["date"]).dt.tz_localize(None)

    market_df.set_index("date", inplace=True)
    onchain_df.set_index("date", inplace=True)

    merged_df = market_df.join(
        onchain_df, how="inner", lsuffix="_market", rsuffix="_onchain"
    )
    merged_df.sort_index(inplace=True)
    merged_df.ffill(inplace=True)

    return merged_df


if __name__ == "__main__":
    from data_collection.market_api import fetch_ohlcv
    from data_collection.on_chain_api import fetch_onchain_metrics

    market_df = fetch_ohlcv("BTC", market="USD")
    onchain_df = fetch_onchain_metrics(
        asset="btc",
        metrics=["AdrActCnt", "TxCnt", "FeeTotNtv"],
        start=market_df["date"].min().strftime("%Y-%m-%d"),
        end=market_df["date"].max().strftime("%Y-%m-%d"),
    )

    combined_df = align_and_merge_data(market_df, onchain_df)

    # print the preview of the combined DataFrame
    print("Combined DataFrame preview:")
    print(combined_df.head())

    output_path = "/Users/komronvalijonov/work/personal/uni-research/data/processed/btc_combined.csv"
    combined_df.to_csv(output_path)
    print(f"Saved merged dataset to {output_path}")
