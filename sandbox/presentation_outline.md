
# Presentation Outline: Bitcoin Price Prediction

**Title:** Bitcoin Price Prediction using a Hybrid CNN-LSTM Model with Attention

**Presenter:** Komron Valijonov

--- 

### Slide 1: Title Slide

*   **Title:** Bitcoin Price Prediction using a Hybrid CNN-LSTM Model with Attention
*   **Subtitle:** An Individual Project for SS25 - MLSS C
*   **Presenter:** Komron Valijonov
*   **Date:** July 8, 2025

---

### Slide 2: Introduction & Motivation

*   **Problem:** High volatility and complexity of the Bitcoin market.
*   **Why it matters:** Accurate prediction is crucial for traders, investors, and financial institutions.
*   **Objective:** To develop a robust deep learning model for Bitcoin price prediction.
*   **Visual:** A chart showing Bitcoin's price volatility over time.

---

### Slide 3: The Dataset

*   **Data Sources:** Market data (OHLCV), on-chain metrics, and technical indicators.
*   **Time Period:** August 17, 2024, to June 28, 2025.
*   **Features:** 18 features in total, including SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and lagged returns.
*   **Visual:** A table summarizing the features used in the model.

---

### Slide 4: Methodology - An Iterative Approach

*   **Model 1: Baseline CNN-LSTM:** A simple hybrid model to establish a performance baseline.
*   **Model 2: CNN-LSTM with Additional Features:** Enhanced with ATR and lagged returns.
*   **Model 3: CNN-LSTM with Attention:** The final model, incorporating feature and temporal attention.
*   **Visual:** A workflow diagram illustrating the iterative model development process.

---

### Slide 5: Model Architecture - CNN-LSTM with Attention

*   **CNN Layer:** Extracts features from the input data.
*   **LSTM Layer:** Captures temporal dependencies.
*   **Attention Mechanism:** Weighs the importance of different features and time steps.
*   **Visual:** A diagram of the CNN-LSTM with Attention architecture.

---

### Slide 6: Results - Model Performance

*   **Comparison:** A table comparing the RMSE and MAE of the three models.
*   **Key Finding:** The CNN-LSTM with Attention model outperforms the other models.
*   **Visual:** The performance comparison table and a bar chart visualizing the RMSE of each model.

---

### Slide 7: Results - Predictions vs. Actuals

*   **Visualization:** A plot showing the model's predictions against the actual Bitcoin prices on the test set.
*   **Analysis:** The model captures the general trend but struggles with high volatility.
*   **Visual:** The `demo_predictions.png` image.

---

### Slide 8: Results - Error Analysis

*   **Distribution:** A histogram of the prediction errors.
*   **Insight:** The errors are centered around zero, but with some large outliers.
*   **Visual:** The `demo_error_distribution.png` image.

---

### Slide 9: Interpretability - Feature Attention

*   **What it is:** The model's ability to weigh the importance of different features.
*   **Key Features:** Bollinger Bands, RSI, MACD, and total transaction fees are the most influential.
*   **Visual:** The `demo_feature_attention.png` image.

---

### Slide 10: Interpretability - Temporal Attention

*   **What it is:** The model's ability to focus on specific time steps in the input sequence.
*   **Insight:** The model pays more attention to more recent time steps.
*   **Visual:** The `demo_temporal_attention.png` image.

---

### Slide 11: Conclusion & Future Work

*   **Summary:** The CNN-LSTM with Attention model is a promising approach for Bitcoin price prediction.
*   **Future Work:**
    *   Incorporate more data sources (e.g., sentiment analysis).
    *   Explore more advanced architectures (e.g., Transformers).
    *   Deploy the model for real-time prediction.

---

### Slide 12: Thank You & Q&A

*   **Contact Information:** Komron Valijonov (Your Email/LinkedIn)
*   **GitHub Repository:** (Link to your project repository)

