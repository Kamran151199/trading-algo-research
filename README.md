# Cryptocurrency Price Prediction with CNN-LSTM and Attention

A comprehensive research project implementing hybrid CNN-LSTM models with attention mechanisms for Bitcoin price prediction. This project demonstrates advanced deep learning techniques applied to cryptocurrency forecasting with real market data.

## 🎯 Project Overview

- **Objective**: Predict Bitcoin prices using hybrid CNN-LSTM models with attention mechanisms
- **Dataset**: 316 days of real Bitcoin data (Aug 2024 - June 2025) with 16 features
- **Best Result**: 12% improvement over baseline with improved attention model
- **RMSE**: $3,295.88 (vs $3,744.17 baseline)

## 📊 Key Results

| Model | RMSE ($) | MAPE (%) | Status |
|-------|----------|----------|---------|
| **Improved Attention v2** | **3,295.88** | **2.62** | 🏆 Best |
| Baseline CNN-LSTM | 3,744.17 | 3.07 | Reference |
| Original Attention | 6,863.91 | 6.08 | ❌ Failed |

## 🏗️ Project Structure

```
├── src/                              # Core project engine
│   ├── models/                       # Neural network architectures
│   │   ├── cnn_lstm.py              # Baseline CNN-LSTM model
│   │   ├── cnn_lstm_with_attention.py # Original attention model
│   │   └── improved_cnn_lstm_attention.py # Improved attention model
│   ├── training/                     # Training pipeline
│   │   └── trainer.py               # Model training framework
│   ├── evaluation/                   # Evaluation metrics
│   │   └── metrics.py               # Performance evaluation
│   ├── preprocessing/                # Data preprocessing
│   │   └── dataset_builder.py       # Data pipeline
│   └── features/                     # Feature engineering
│       └── technical_features.py    # Technical indicators
├── data/                            # Dataset files
│   ├── raw/                         # Raw market data
│   └── processed/                   # Preprocessed data
├── notebooks/                       # Jupyter demonstrations
│   ├── 02_cnn_lstm.ipynb           # Baseline model demo
│   └── 03_cnn_lstm_with_attention.ipynb # Attention model demo
├── figures/                         # Generated visualizations
├── research_report.md               # Complete research documentation
└── pyproject.toml                  # Project dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd uni-research

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Run Demonstrations

**Option A: Jupyter Notebooks (Recommended)**
```bash
# Start Jupyter
jupyter lab

# Open and run:
# - notebooks/02_cnn_lstm.ipynb (Baseline model)
# - notebooks/03_cnn_lstm_with_attention.ipynb (Attention model)
```

**Option B: Python Scripts**
```bash
# Train baseline model
python -m src.training.trainer

# Evaluate models
python -m src.evaluation.metrics
```

### 3. View Results

- **Research Report**: Open `research_report.md` for complete analysis
- **Visualizations**: Check `figures/` directory for all charts
- **Model Outputs**: Find predictions in `notebooks/bitcoin_predictions_*.csv`

## 📈 Data Features

**Market Data (5 features):**
- Open, High, Low, Close, Volume

**On-chain Metrics (3 features):**
- Active addresses (adractcnt)
- Transaction count (txcnt) 
- Fee data (feetotntv)

**Technical Indicators (8 features):**
- Simple Moving Average (SMA), Exponential Moving Average (EMA)
- Relative Strength Index (RSI), MACD, Bollinger Bands

## 🔬 Model Architectures

### Baseline CNN-LSTM
- 1D CNN for local pattern extraction
- LSTM for temporal dependencies
- Parameters: 110,913

### Improved Attention Model
- Simplified single-head temporal attention
- Enhanced regularization (dropout 0.4)
- Better weight initialization
- Parameters: 119,234

## 📊 Key Findings

1. **Attention Success**: Properly designed attention mechanisms improve prediction accuracy by 12%
2. **Simplicity Wins**: Simplified attention outperforms complex multi-head variants
3. **Regularization Critical**: High dropout (0.4) essential for small datasets
4. **Initialization Matters**: Xavier/Kaiming initialization prevents training issues

## 🎯 Research Contributions

- Demonstrated successful attention mechanism improvement from failure to success
- Provided guidelines for attention model design on small datasets
- Created reusable framework for cryptocurrency price prediction
- Established best practices for CNN-LSTM hybrid architectures

## 📋 Requirements

- Python 3.9+
- PyTorch 2.5.1
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- jupyter

See `pyproject.toml` for complete dependency list.

## 🎓 Academic Usage

This project is designed for academic research and education. Key components:

- **Reproducible**: Fixed random seeds, documented procedures
- **Well-documented**: Comprehensive code comments and README
- **Modular**: Clean separation of concerns in src/ directory
- **Validated**: Real data, actual results, no simulation

## 📚 Documentation

- **`research_report.md`**: Complete research documentation with methodology, results, and analysis
- **`notebooks/`**: Interactive demonstrations of model training and evaluation
- **`src/`**: Documented source code with type hints and comments

## ⚠️ Important Notes

- **Real Data Only**: All results based on actual Bitcoin market data
- **No Simulation**: No fake or simulated data used
- **Production Ready**: Models can be deployed for real trading (use at your own risk)
- **Educational Purpose**: Designed for learning and research

## 🤝 Contributing

This is an academic research project. For educational use and research extensions.

## 📄 License

Academic use only. Please cite if used in research.

---

**Success Story**: This project transformed a failed attention model (83% worse than baseline) into a winning model (12% better than baseline) through systematic analysis and improvement. See `research_report.md` for the complete journey.
