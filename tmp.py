import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats


class BitcoinCLTTester:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_bitcoin_data(self, symbol="BTC", market="USD", interval="1min"):
        """
        Fetch Bitcoin data from Alpha Vantage
        """
        params = {
            "function": "CRYPTO_INTRADAY",
            "symbol": symbol,
            "market": market,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": "full",
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            # Check for different possible time series keys
            time_series_key = None
            possible_keys = ["Time Series (Crypto)", "Time Series Crypto (1min)"]

            for key in possible_keys:
                if key in data:
                    time_series_key = key
                    break

            if time_series_key:
                df = pd.DataFrame(data[time_series_key]).T
                df.index = pd.to_datetime(df.index)

                # The API returns columns with numbered prefixes, so we need to rename them
                column_mapping = {
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. volume": "volume",
                }
                df = df.rename(columns=column_mapping)
                df = df.astype(float)
                return df.sort_index()
            else:
                print("Error in API response:", data)
                return None

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def calculate_returns(self, df):
        """
        Calculate 1-minute returns
        """
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        return df.dropna()

    def apply_clt_analysis(self, returns, sample_size=10, num_samples=15):
        """
        Apply Central Limit Theorem analysis
        """
        if len(returns) < sample_size * num_samples:
            print(
                f"Not enough data. Need {sample_size * num_samples}, got {len(returns)}"
            )
            return None

        # Create samples and calculate means
        sample_means = []
        for i in range(num_samples):
            start_idx = i * sample_size
            end_idx = start_idx + sample_size
            if end_idx <= len(returns):
                sample = returns.iloc[start_idx:end_idx]
                sample_means.append(sample.mean())

        sample_means = np.array(sample_means)

        # Calculate statistics
        mean_of_means = np.mean(sample_means)
        std_of_means = np.std(sample_means)

        # Test for normality
        _, p_value = stats.shapiro(sample_means)

        return {
            "sample_means": sample_means,
            "mean": mean_of_means,
            "std": std_of_means,
            "normality_p_value": p_value,
            "is_normal": p_value > 0.05,
        }

    def predict_price_ranges(
        self, current_price, clt_results, confidence_levels=[0.68, 0.95]
    ):
        """
        Predict price movement ranges based on CLT results
        """
        predictions = {}

        for confidence in confidence_levels:
            # Calculate z-score for confidence level
            z_score = stats.norm.ppf((1 + confidence) / 2)

            # Calculate expected return range
            margin_of_error = z_score * clt_results["std"]

            # Convert to price range
            lower_return = clt_results["mean"] - margin_of_error
            upper_return = clt_results["mean"] + margin_of_error

            lower_price = current_price * (1 + lower_return)
            upper_price = current_price * (1 + upper_return)

            predictions[confidence] = {
                "lower_price": lower_price,
                "upper_price": upper_price,
                "range": upper_price - lower_price,
                "return_range": (lower_return, upper_return),
            }

        return predictions

    def backtest_predictions(self, df, sample_size=10, prediction_window=10):
        """
        Backtest the CLT predictions
        """
        results = []

        # Need enough data for analysis + prediction
        min_data_needed = sample_size * 20  # 20 samples minimum

        for i in range(min_data_needed, len(df) - prediction_window, prediction_window):
            # Get historical data for CLT analysis
            hist_returns = df["returns"].iloc[i - min_data_needed : i]

            # Apply CLT
            clt_results = self.apply_clt_analysis(hist_returns, sample_size, 20)

            if clt_results is None:
                continue

            # Current price
            current_price = df["close"].iloc[i]

            # Make predictions
            predictions = self.predict_price_ranges(current_price, clt_results)

            # Check actual price after prediction window
            actual_price = df["close"].iloc[i + prediction_window]
            actual_return = (actual_price - current_price) / current_price

            # Check if predictions were correct
            test_result = {
                "timestamp": df.index[i],
                "current_price": current_price,
                "actual_price": actual_price,
                "actual_return": actual_return,
                "predictions": predictions,
                "clt_mean": clt_results["mean"],
                "clt_std": clt_results["std"],
                "is_normal": clt_results["is_normal"],
            }

            # Check accuracy
            for confidence in predictions:
                pred = predictions[confidence]
                test_result[f"hit_{confidence}"] = (
                    pred["lower_price"] <= actual_price <= pred["upper_price"]
                )

            results.append(test_result)

        return results

    def analyze_results(self, backtest_results):
        """
        Analyze backtest results
        """
        if not backtest_results:
            print("No backtest results to analyze")
            return

        df_results = pd.DataFrame(backtest_results)

        print("=== BACKTEST RESULTS ===")
        print(f"Total predictions: {len(df_results)}")
        print(
            f"Predictions where CLT assumptions held (normal): {df_results['is_normal'].sum()}"
        )

        # Accuracy by confidence level
        for confidence in [0.68, 0.95]:
            if f"hit_{confidence}" in df_results.columns:
                accuracy = df_results[f"hit_{confidence}"].mean()
                print(
                    f"{confidence * 100:.0f}% Confidence Interval Accuracy: {accuracy:.2%}"
                )

        # Statistics
        print("\nActual Return Statistics:")
        print(f"Mean: {df_results['actual_return'].mean():.4f}")
        print(f"Std: {df_results['actual_return'].std():.4f}")
        print(f"Min: {df_results['actual_return'].min():.4f}")
        print(f"Max: {df_results['actual_return'].max():.4f}")

        return df_results

    def plot_results(self, df_results):
        """
        Plot analysis results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Actual vs Predicted Returns
        axes[0, 0].scatter(
            df_results["clt_mean"], df_results["actual_return"], alpha=0.6
        )
        axes[0, 0].plot([-0.01, 0.01], [-0.01, 0.01], "r--", label="Perfect prediction")
        axes[0, 0].set_xlabel("CLT Predicted Mean Return")
        axes[0, 0].set_ylabel("Actual Return")
        axes[0, 0].set_title("Predicted vs Actual Returns")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot 2: Histogram of actual returns
        axes[0, 1].hist(df_results["actual_return"], bins=30, alpha=0.7, density=True)
        axes[0, 1].set_xlabel("Actual Returns")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].set_title("Distribution of Actual Returns")
        axes[0, 1].grid(True)

        # Plot 3: Accuracy over time
        df_results["accuracy_68"] = df_results["hit_0.68"].rolling(window=20).mean()
        df_results["accuracy_95"] = df_results["hit_0.95"].rolling(window=20).mean()

        axes[1, 0].plot(df_results["accuracy_68"], label="68% CI Accuracy")
        axes[1, 0].plot(df_results["accuracy_95"], label="95% CI Accuracy")
        axes[1, 0].axhline(y=0.68, color="blue", linestyle="--", alpha=0.5)
        axes[1, 0].axhline(y=0.95, color="orange", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Prediction Number")
        axes[1, 0].set_ylabel("Rolling Accuracy")
        axes[1, 0].set_title("Prediction Accuracy Over Time")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot 4: CLT assumption validity
        normal_count = df_results.groupby("is_normal").size()
        axes[1, 1].bar(["Non-Normal", "Normal"], normal_count.values)
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("CLT Assumption Validity")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()


def main():
    # Initialize with your API key
    API_KEY = "34G52AJX6YNA023L"  # Replace with your actual API key

    tester = BitcoinCLTTester(API_KEY)

    print("Fetching Bitcoin data...")
    df = tester.get_bitcoin_data()

    if df is None:
        print("Failed to fetch data. Please check your API key and connection.")
        return

    print(f"Fetched {len(df)} data points")
    print(f"Data range: {df.index.min()} to {df.index.max()}")

    # Calculate returns
    df = tester.calculate_returns(df)
    print(f"Calculated returns for {len(df)} periods")

    # Run backtest
    print("\nRunning backtest...")
    results = tester.backtest_predictions(df, sample_size=10, prediction_window=10)

    if results:
        # Analyze results
        df_results = tester.analyze_results(results)

        # Plot results
        tester.plot_results(df_results)

        # Example: Make a live prediction
        print("\n=== LIVE PREDICTION EXAMPLE ===")
        recent_returns = df["returns"].tail(600)  # Last 600 minutes
        clt_results = tester.apply_clt_analysis(recent_returns, 30, 20)

        if clt_results:
            current_price = df["close"].iloc[-1]
            predictions = tester.predict_price_ranges(current_price, clt_results)

            print(f"Current BTC Price: ${current_price:.2f}")
            print(f"CLT Mean Return: {clt_results['mean']:.4f}")
            print(f"CLT Std: {clt_results['std']:.4f}")
            print(f"Normal distribution: {clt_results['is_normal']}")

            for confidence, pred in predictions.items():
                print(f"\n{confidence * 100:.0f}% Confidence Interval:")
                print(
                    f"  Price range: ${pred['lower_price']:.2f} - ${pred['upper_price']:.2f}"
                )
                print(f"  Range width: ${pred['range']:.2f}")

    else:
        print("No backtest results generated. Check if you have enough data.")


if __name__ == "__main__":
    main()
