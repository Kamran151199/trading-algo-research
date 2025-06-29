# SS25 - Machine Learning C - Project Proposal

## Table of Contents
1. [Introduction and Problem Statement](#introduction-and-problem-statement)
   - [Problem Statement](#problem-statement)
   - [Why Is It Interesting to Work On?](#why-is-it-interesting-to-work-on)
   - [Real-World Applications](#real-world-applications)
2. [Literature Review and Gap Analysis](#literature-review-and-gap-analysis)
   - [What has been done so far in the field?](#what-has-been-done-so-far-in-the-field)
   - [What is remaining to be done (Gap Analysis)](#what-is-remaining-to-be-done-gap-analysis)
3. [Your novelty and contribution](#your-novelty-and-contribution)
   - [Which 5 interesting research questions are you going to investigate?](#which-5-interesting-research-questions-are-you-going-to-investigate)
   - [What is novelty of these questions?](#what-is-novelty-of-these-questions)
4. [Your methodology](#your-methodology)
   - [How you will address these research questions?](#how-you-will-address-these-research-questions)
   - [What is your intended dataset on which you will work?](#what-is-your-intended-dataset-on-which-you-will-work-give-its-kaggle-link)
   - [Which deep learning model will you deploy for this task?](#which-deep-learning-model-will-you-deploy-for-this-task)
   - [What is the main methodology you will perform your experiments?](#what-is-the-main-methodology-you-will-perform-your-experiments-draw-a-simple-workflow-diagram-for-answering-this-part)
5. [Your expected results](#your-expected-results)
   - [How your results will look like?](#how-your-results-will-look-like-what-kind-of-figures-tables-and-equations-are-you-going-to-have-as-answers-to-each-of-your-research-questions)

## Introduction and Problem Statement

### Problem Statement

- The financial markets, particularly cryptocurrency and stock trading, generate massive amounts of high-dimensional, non-linear, and non-stationary time-series data that traditional statistical models struggle to analyze effectively.
- Market inefficiencies and price discrepancies across different exchanges and trading pairs create opportunities for algorithmic trading strategies that human traders cannot capitalize on due to reaction time limitations.
- The high volatility and unpredictable nature of cryptocurrency markets pose significant challenges for accurate price prediction and risk management, requiring advanced machine learning techniques.
- Existing trading algorithms often fail to adapt to rapidly changing market conditions, leading to deteriorating performance over time as market dynamics evolve.
- The integration of alternative data sources (social media sentiment, on-chain metrics, macroeconomic indicators) with traditional price and volume data remains a complex challenge that requires sophisticated feature engineering and multi-modal learning approaches.
- Market manipulation and fraudulent activities in cryptocurrency markets create noise in the data that can mislead predictive models, necessitating robust anomaly detection mechanisms.
- The lack of standardized evaluation metrics for trading algorithms makes it difficult to compare different approaches objectively, highlighting the need for comprehensive performance frameworks that consider risk-adjusted returns, drawdowns, and consistency.
- The computational efficiency of trading algorithms is critical for real-time decision-making, yet many sophisticated machine learning models are too slow for high-frequency trading applications.

### Why Is It Interesting to Work On?

- The intersection of finance and machine learning creates a perfect testing ground for reinforcement learning algorithms, where agents can learn optimal trading strategies through direct interaction with market environments.
- The financial markets provide immediate and quantifiable feedback on model performance through profit and loss metrics, allowing for rapid iteration and improvement of algorithms.
- The non-stationary nature of financial markets challenges traditional machine learning assumptions about data distribution, pushing researchers to develop novel approaches for continual learning and adaptation.
- The high stakes of financial trading create strong incentives for developing robust and reliable models, with clear economic value for successful implementations.
- The availability of vast amounts of historical and real-time market data provides rich opportunities for developing and testing complex deep learning architectures.
- The multi-faceted nature of market analysis (technical indicators, fundamental analysis, sentiment analysis) allows for creative integration of different machine learning paradigms.
- The practical application of theoretical concepts in reinforcement learning, time series forecasting, and natural language processing makes this domain intellectually stimulating and commercially relevant.
- The emergence of decentralized finance (DeFi) introduces new trading mechanisms and market structures that have not been extensively studied from a machine learning perspective.

### Real-World Applications

- Automated trading systems that can execute trades 24/7 across global markets without human intervention, potentially capturing opportunities that would be missed by manual trading.
- Portfolio optimization algorithms that dynamically adjust asset allocations based on changing market conditions and risk preferences.
- Risk management tools that can predict market volatility spikes and potential drawdowns, allowing traders to adjust their exposure accordingly.
- Market making strategies that provide liquidity to exchanges while managing inventory risk through intelligent order placement and execution.
- Arbitrage detection systems that identify and exploit price discrepancies across different exchanges or related assets.
- Sentiment analysis tools that monitor social media, news, and forum discussions to gauge market sentiment and predict potential price movements.
- Anomaly detection systems that identify unusual market behavior that might indicate manipulation or insider trading.
- Personalized trading assistants that provide recommendations tailored to individual trader preferences, risk tolerance, and investment goals.
- Regulatory technology (RegTech) solutions that help trading firms comply with complex financial regulations while maintaining trading efficiency.

## Literature Review and Gap Analysis

### What has been done so far in the field?

- Cohen & Aiche (2025) developed an AI-driven strategy for Bitcoin price prediction that achieved a 1640.32% return by leveraging an ensemble of neural networks and incorporating social media sentiment data alongside traditional financial metrics.
- Rodrigues & Machado (2025) conducted a comparative study of machine learning models for high-frequency cryptocurrency forecasting, finding that GRU neural networks demonstrated superior predictive accuracy with a MAPE of 0.09% for 60-minute ahead predictions.
- Gurgul et al. (2025) introduced novel approaches to cryptocurrency price forecasting by integrating deep learning NLP techniques to analyze social media content, demonstrating that local extrema are a valid alternative to daily price changes as predictive targets.
- Kehinde et al. (2025) developed the Helformer model, which integrates Holt-Winters exponential smoothing with Transformer-based deep learning architecture to decompose cryptocurrency time series data into level, trend, and seasonality components, enhancing prediction accuracy.
- Kumar & Ji (2025) proposed CryptoPulse, a dual prediction mechanism that forecasts next-day cryptocurrency closing prices by incorporating macroeconomic fluctuations, technical indicators, and market sentiment-based rescaling and fusion.
- Balijepalli & Thangaraj (2025) created a dynamic forecasting model using ensemble machine learning methods to test the forecasting accuracy of top 15 cryptocurrencies' prices, demonstrating that ensemble approaches outperform standalone models.
- Pereira (2025) analyzed how AI is disrupting stock markets through algorithmic trading, sentiment analysis, and personalized investment advice, highlighting how AI image recognition can identify investment opportunities from visual cues in the environment.
- Sanchez (2023) applied various machine learning techniques for quantitative finance in cryptocurrency markets, providing a comprehensive framework for cryptocurrency trading and investment strategies.

### What is remaining to be done (Gap Analysis)

- Current research predominantly focuses on major cryptocurrencies like Bitcoin and Ethereum, with limited exploration of altcoins and emerging digital assets that may exhibit different price dynamics and volatility patterns.
- Most existing models fail to adequately integrate on-chain metrics (transaction volume, active addresses, mining difficulty) with traditional price data and sentiment analysis, missing crucial blockchain-specific indicators that could improve prediction accuracy.
- There is insufficient research on the interpretability and explainability of deep learning models in cryptocurrency trading, making it difficult for traders to trust and understand the reasoning behind AI-generated predictions.
- The impact of regulatory announcements and geopolitical events on cryptocurrency markets is not systematically incorporated into most prediction models, despite their significant influence on market volatility.
- Few studies have explored the application of reinforcement learning for adaptive trading strategies that can evolve with changing market conditions and learn from their own trading decisions in real-time.

## Your novelty and contribution

### Which 5 interesting research questions are you going to investigate?

1. How can a hybrid model integrating on-chain metrics, technical indicators, and NLP-based sentiment analysis improve the accuracy of cryptocurrency price predictions compared to models using only historical price data?
2. To what extent can reinforcement learning algorithms develop adaptive trading strategies that outperform traditional machine learning approaches in volatile cryptocurrency markets?
3. How does the predictive performance of deep learning models differ when forecasting price movements of major cryptocurrencies versus emerging altcoins with lower market capitalization?
4. What impact do regulatory announcements and geopolitical events have on cryptocurrency price predictions, and how can these factors be systematically incorporated into forecasting models?
5. How can explainable AI techniques be applied to deep learning models for cryptocurrency trading to increase transparency and trust while maintaining predictive accuracy?

### What is novelty of these questions?

The novelty of these research questions lies in their comprehensive approach to addressing critical gaps in current cryptocurrency forecasting research:

**Integration of Multi-Modal Data Sources:** While existing research typically focuses on either price data, sentiment analysis, or technical indicators in isolation, our first research question proposes a novel hybrid approach that integrates all three data types along with blockchain-specific on-chain metrics. This holistic approach acknowledges the unique nature of cryptocurrency markets where blockchain activity provides additional signals not available in traditional financial markets.

**Adaptive Learning Through Reinforcement:** Most current models use supervised learning approaches that struggle to adapt to rapidly changing market conditions. Our second question explores reinforcement learning's potential to create self-improving trading strategies that learn from their own decisions and market feedback, representing a paradigm shift from static to dynamic prediction models.

**Market Capitalization Diversity:** Existing research disproportionately focuses on Bitcoin and Ethereum, neglecting the diverse ecosystem of cryptocurrencies. Our third question addresses this limitation by comparing model performance across cryptocurrencies with varying market capitalizations, potentially uncovering unique patterns and relationships specific to emerging digital assets.

**Event-Driven Market Analysis:** While many researchers acknowledge the impact of external events on cryptocurrency markets, few have systematically incorporated these factors into their models. Our fourth question proposes a structured approach to quantifying and integrating regulatory and geopolitical events into prediction frameworks, potentially improving model robustness during periods of external market shocks.

**Explainable AI for Financial Decision-Making:** The "black box" nature of deep learning models has limited their adoption in financial decision-making where transparency is crucial. Our fifth question explores the novel application of explainable AI techniques specifically tailored for cryptocurrency trading models, potentially bridging the gap between model complexity and interpretability without sacrificing performance.

These research questions collectively represent a significant advancement in the field by addressing fundamental limitations in current approaches while exploring new methodologies that could transform how cryptocurrency price predictions are generated and utilized by traders, investors, and researchers.

## Your methodology

### How you will address these research questions?

Our methodology employs a comprehensive approach to address the research questions through a combination of data collection, preprocessing, model development, and evaluation. The research will be conducted in several interconnected phases:

#### Dataset Selection

For this study, we will utilize the Cryptocurrency Historical Prices dataset from Kaggle, which contains historical price data for over 100 cryptocurrencies from 2013 to present. This dataset includes daily open, high, low, close prices, volume, and market capitalization information. We will supplement this with:

- **On-chain metrics:** Transaction counts, active addresses, mining difficulty, and hash rates obtained through blockchain APIs
- **Social media sentiment data:** Twitter and Reddit posts related to cryptocurrencies
- **News articles:** Financial news from major sources covering cryptocurrency markets
- **Regulatory announcements:** Official statements from regulatory bodies worldwide
- **Technical indicators:** RSI, MACD, Bollinger Bands, and other common trading indicators

#### Data Preprocessing

- **Time series alignment:** Synchronizing data from different sources to create a unified timeline
- **Missing value imputation:** Using advanced techniques like MICE (Multivariate Imputation by Chained Equations)
- **Feature engineering:** Creating derived features from raw data, including volatility measures, momentum indicators, and sentiment scores
- **Normalization:** Scaling features to comparable ranges using techniques like min-max scaling or standardization
- **Temporal splitting:** Dividing data into training (70%), validation (15%), and testing (15%) sets with careful consideration of temporal dependencies

#### Model Development

We will develop and compare several deep learning architectures:

- **Hybrid CNN-LSTM Network:** Combining convolutional layers for feature extraction with LSTM layers for temporal pattern recognition
- **Transformer-based Model:** Utilizing attention mechanisms to capture long-range dependencies in time series data
- **Reinforcement Learning Agent:** Implementing Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms for adaptive trading strategy development
- **Explainable AI Framework:** Integrating SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to provide interpretability

#### Evaluation Metrics

The models will be evaluated using multiple metrics to ensure comprehensive performance assessment:

- **Prediction Accuracy:** RMSE, MAE, MAPE for regression tasks; Accuracy, F1-score for classification tasks
- **Financial Performance:** Sharpe ratio, maximum drawdown, total return, and win rate in simulated trading
- **Adaptability:** Performance stability across different market conditions (bull, bear, and sideways markets)
- **Explainability:** Quantitative measures of feature importance and decision transparency

#### Experimental Design

For each research question, we will design specific experiments:

- **RQ1 (Hybrid Model):** Compare performance of models with different combinations of data sources (price-only, price+sentiment, price+on-chain, and all-inclusive)
- **RQ2 (Reinforcement Learning):** Evaluate RL agents against traditional ML approaches across various market conditions
- **RQ3 (Market Cap Diversity):** Test models on cryptocurrencies grouped by market capitalization (large, medium, small)
- **RQ4 (Event Impact):** Measure prediction errors before, during, and after significant regulatory announcements
- **RQ5 (Explainability):** Compare user trust and decision confidence when using black-box versus explainable models

### What is your intended dataset on which you will work? Give its Kaggle link.

We will use the "Cryptocurrency Historical Prices" dataset from Kaggle, available at: https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrency-historical-prices-data

This comprehensive dataset includes historical price data for over 100 cryptocurrencies, with daily OHLCV (Open, High, Low, Close, Volume) data and market capitalization information. The dataset is regularly updated and covers the period from 2013 to present, providing a rich historical context for our analysis.

### Which deep learning model will you deploy for this task?

For this research, we will deploy multiple deep learning models to address different aspects of our research questions:

**Primary Model:** A hybrid architecture combining CNN-LSTM networks with attention mechanisms. The CNN layers will extract features from multivariate time series data, while the LSTM layers will capture temporal dependencies. Attention mechanisms will help the model focus on the most relevant time steps and features.

**Secondary Models:**
- **Transformer-based model:** Leveraging the self-attention mechanism to capture long-range dependencies in time series data without the sequential constraints of RNNs.
- **Deep Reinforcement Learning:** Implementing Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO) for adaptive trading strategy development.
- **BERT-based models:** For processing and analyzing textual data from news articles and social media to extract sentiment and event information.

**Explainability Layer:** We will integrate SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) frameworks to provide interpretability for our deep learning models.

These models will be implemented using TensorFlow 2.x and PyTorch frameworks, with hyperparameter optimization performed using Bayesian optimization techniques.

### What is the main methodology you will perform your experiments? Draw a simple workflow diagram for answering this part.

Our experimental methodology follows a systematic approach that integrates data from multiple sources, applies sophisticated preprocessing techniques, develops and trains various deep learning models, and evaluates their performance using comprehensive metrics.

![Cryptocurrency Price Prediction Workflow Diagram](figures/workflow_diagram.png)

The workflow diagram illustrates our end-to-end methodology, starting with data collection from multiple sources (price data, on-chain metrics, social media sentiment, and news/regulatory data), followed by preprocessing steps including time series alignment, feature engineering, and NLP processing. These preprocessed data streams are then integrated into a unified dataset, which feeds into our various model architectures (Hybrid CNN-LSTM, Transformer, Reinforcement Learning, and Explainable AI Framework). Finally, the models are evaluated using multiple metrics including prediction accuracy, financial performance, and adaptability/explainability measures.

## Your expected results

### How your results will look like? What kind of figures, tables, and equations are you going to have as answers to each of your research questions?

This section outlines the anticipated results for each research question, including dummy figures and tables that illustrate the expected findings.

#### Research Question 1: Hybrid Model Integration

We expect that our hybrid model integrating on-chain metrics, technical indicators, and NLP-based sentiment analysis will significantly outperform models using only historical price data. The anticipated results will demonstrate:

1. Lower prediction errors (RMSE and MAE) for the hybrid model compared to single-source models
2. Higher risk-adjusted returns (Sharpe ratio) for trading strategies based on the hybrid model
3. More stable performance across different market conditions

![Model Comparison](figures/expected_results/rq1_model_comparison.png)

*Figure 1: Comparison of prediction errors and risk-adjusted returns across different model types. The hybrid model shows superior performance in both accuracy metrics and financial performance.*

| Model Type | RMSE | MAE | Sharpe Ratio | Max Drawdown (%) | Annual Return (%) |
|------------|------|-----|--------------|------------------|-------------------|
| Price-Only | 0.0245 | 0.0198 | 1.2 | 28.4 | 32.7 |
| Price+Sentiment | 0.0187 | 0.0152 | 1.7 | 22.1 | 41.5 |
| Price+On-Chain | 0.0163 | 0.0131 | 1.9 | 19.8 | 45.2 |
| Hybrid Model | 0.0112 | 0.0089 | 2.4 | 15.3 | 52.8 |

*Table 1: Performance metrics for different model types, showing the hybrid model's superior performance across all evaluation criteria.*

#### Research Question 2: Reinforcement Learning for Adaptive Trading

We anticipate that reinforcement learning algorithms will develop adaptive trading strategies that outperform traditional machine learning approaches, particularly during periods of high volatility. The expected results will show:

1. Higher cumulative returns for adaptive RL strategies compared to static ML models
2. Better recovery from market downturns
3. Improved performance consistency across different market regimes

![RL Comparison](figures/expected_results/rq2_rl_comparison.png)

*Figure 2: Cumulative returns over time for traditional ML, basic RL, and adaptive RL trading strategies. The adaptive RL strategy demonstrates superior performance, especially during market volatility.*

| Strategy Type | Cumulative Return (%) | Sharpe Ratio | Max Drawdown (%) | Win Rate (%) | Avg. Trade Duration |
|---------------|------------------------|--------------|------------------|--------------|---------------------|
| Traditional ML | 42.3 | 1.4 | 24.7 | 58.2 | 3.2 days |
| Basic RL | 67.8 | 1.8 | 21.2 | 62.5 | 2.8 days |
| Adaptive RL | 89.5 | 2.2 | 18.5 | 65.7 | 2.5 days |

*Table 2: Performance comparison of different trading strategy types, highlighting the adaptive RL strategy's superior risk-adjusted returns and reduced drawdowns.*

#### Research Question 3: Market Cap Diversity

We expect to find significant differences in the predictive performance of deep learning models when forecasting price movements across cryptocurrencies with varying market capitalizations. The anticipated results will reveal:

1. Higher prediction accuracy for large-cap cryptocurrencies
2. Greater profit potential but higher risk for small-cap cryptocurrencies
3. Different optimal feature sets for each market cap category

![Market Cap Diversity](figures/expected_results/rq3_market_cap_diversity.png)

*Figure 3: Prediction performance and risk-return profiles across cryptocurrency market cap categories. While large-cap cryptocurrencies show higher prediction accuracy, small-cap cryptocurrencies offer higher potential returns with increased risk.*

| Market Cap Category | Prediction Accuracy (%) | F1 Score | Profit Potential (%) | Risk (Volatility %) | Key Predictive Features |
|--------------------|-------------------------|----------|----------------------|---------------------|-------------------------|
| Large Cap | 78.2 | 0.76 | 22.4 | 12.3 | Price momentum, Trading volume, Social sentiment |
| Medium Cap | 72.5 | 0.69 | 28.7 | 18.5 | Social sentiment, On-chain metrics, Exchange flows |
| Small Cap | 65.1 | 0.61 | 35.2 | 24.1 | Social sentiment, Developer activity, Token economics |

*Table 3: Performance metrics and key predictive features across different market capitalization categories.*

#### Research Question 4: Regulatory and Geopolitical Event Impact

We anticipate that regulatory announcements and geopolitical events significantly impact cryptocurrency price predictions. The expected results will demonstrate:

1. Increased prediction errors during and immediately after significant events
2. Varying impact magnitudes based on event type
3. Improved prediction accuracy when event data is incorporated into the model

![Event Impact](figures/expected_results/rq4_event_impact.png)

*Figure 4: Impact of different event types on prediction error before, during, and after the event. Market crashes and exchange hacks show the most significant impact on prediction accuracy.*

| Event Type | Pre-Event Error | During-Event Error | Post-Event Error | Recovery Time | Optimal Model Adjustment |
|------------|-----------------|-------------------|------------------|---------------|--------------------------|
| Regulatory Announcement | 0.021 | 0.045 | 0.028 | 3.2 days | Increase sentiment weight |
| Exchange Hack | 0.018 | 0.062 | 0.035 | 5.7 days | Increase on-chain weight |
| Protocol Upgrade | 0.015 | 0.022 | 0.017 | 1.5 days | Increase technical weight |
| Market Crash | 0.025 | 0.078 | 0.042 | 8.3 days | Decrease price history weight |

*Table 4: Prediction errors and recovery metrics for different event types, with recommended model adjustments for each scenario.*

#### Research Question 5: Explainable AI for Trading Models

We expect that explainable AI techniques will increase transparency and trust in deep learning models for cryptocurrency trading while maintaining predictive accuracy. The anticipated results will show:

1. Key feature importance rankings that align with financial theory
2. Improved user trust and decision confidence with explainable models
3. Minimal trade-off between model interpretability and performance

![Feature Importance](figures/expected_results/rq5_feature_importance.png)

*Figure 5: Relative importance of different features in cryptocurrency price prediction. Price momentum and trading volume emerge as the most influential factors, followed by social sentiment.*

![SHAP Values](figures/expected_results/rq5_shap_values.png)

*Figure 6: SHAP values for sample predictions, showing the impact of each feature on individual predictions. Red points indicate high feature values, while blue points indicate low values.*

| Explainability Method | User Trust Score | Decision Confidence | Performance Impact | Implementation Complexity | Best Use Case |
|-----------------------|------------------|---------------------|-------------------|---------------------------|---------------|
| SHAP | 8.7/10 | 8.5/10 | -2.1% | Medium | Feature importance |
| LIME | 7.9/10 | 7.8/10 | -1.5% | Low | Local explanations |
| Attention Visualization | 8.2/10 | 8.0/10 | -0.8% | High | Temporal patterns |
| Rule Extraction | 9.1/10 | 8.9/10 | -3.7% | Very High | Regulatory compliance |

*Table 5: Comparison of different explainability methods for cryptocurrency trading models, highlighting their impact on user trust, decision confidence, and model performance.*
