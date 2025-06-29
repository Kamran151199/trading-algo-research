"""
Alpha Vantage Market Data & News Downloader

Provides access to historical OHLCV data and news/sentiment via Alpha Vantage API.

Author: Research Team
"""

import logging
import os
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_URL = "https://www.alphavantage.co/query"


def generate_fake_headers():
    """
    Generate fake headers for requests to avoid rate limiting.
    """
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "Pragma": "no-cache",
    }


def fetch_ohlcv(
    symbol: str, market: str = "USD", interval: str = "daily", outputsize: str = "full"
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given cryptocurrency symbol.
    """
    function = (
        "DIGITAL_CURRENCY_DAILY" if interval == "daily" else "DIGITAL_CURRENCY_INTRADAY"
    )

    params = {
        "function": function,
        "symbol": symbol.upper(),
        "market": market.upper(),
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    logger.info(f"Requesting OHLCV data for {symbol}-{market}")
    response = requests.get(BASE_URL, params=params, headers=generate_fake_headers())
    data = response.json()

    key = "Time Series (Digital Currency Daily)"
    if key not in data:
        logger.error(f"No data returned for {symbol}: {data}")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data[key], orient="index")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "date"}, inplace=True)

    # Standardize and rename relevant columns
    rename_map = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume",
        "6. market cap (usd)": "market_cap",
    }

    df.rename(columns=rename_map, inplace=True)

    # Keep only relevant columns
    keep_cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[[col for col in keep_cols if col in df.columns]]

    df = df.astype(
        {
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float",
        }
    )

    df["symbol"] = symbol
    df["source"] = "alpha_vantage"
    df["download_timestamp"] = datetime.now()

    return df


def fetch_news(topics: str = "crypto", limit: int = 100) -> pd.DataFrame:
    """
    Fetch recent news articles and sentiment scores.
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "topics": topics,
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    logger.info(f"Fetching news and sentiment for topics: {topics}")
    response = requests.get(BASE_URL, params=params, headers=generate_fake_headers())
    data = response.json()

    if "feed" not in data:
        logger.error("News data not returned")
        return pd.DataFrame()

    feed = data["feed"]
    df = pd.DataFrame(feed)
    df["download_timestamp"] = datetime.now()

    return df


if __name__ == "__main__":
    # Sample usage for testing
    print("=== Fetching BTC Daily OHLCV ===")
    df_price = fetch_ohlcv("BTC", market="USD")
    print(df_price.head())

    print("\n=== Fetching Crypto News Sentiment ===")
    df_news = fetch_news()
    print(df_news[["title", "summary"]].head())

    # store the data in CSV files
    df_price.to_csv(
        "/Users/komronvalijonov/work/personal/uni-research/data/raw/btc_daily_ohlcv.csv",
        index=False,
    )
    df_news.to_csv(
        "/Users/komronvalijonov/work/personal/uni-research/data/raw/crypto_news_sentiment.csv",
        index=False,
    )
    print("\nData saved to CSV files.")
    logger.info("Data collection completed successfully.")
    print("Data collection completed successfully.")
