# Stock Market Prediction with Machine Learning

A comprehensive guide to machine learning approaches for stock market prediction, from traditional statistical methods to more advanced deep learning techniques. If you are new to the world of predictive modelling in finance, or have always dreamt of predicting the stock market, you've come to the right place!

[![A Danny Blaker project badge](https://github.com/dannyblaker/dannyblaker.github.io/blob/main/danny_blaker_project_badge.svg)](https://github.com/dannyblaker/)

**LIABILITY DISCLAIMER: This project is provided for educational and research purposes only. It does not constitute financial advice, and the author makes no guarantees about the accuracy, completeness, or suitability of the results produced by this code. By using this repository, you agree that the author is not liable for any financial losses or decisions made based on its use.** 

Stock market prediction is extremely challenging. Use these tools responsibly and always validate approaches thoroughly before any real-world application.

## 📚 Overview

This repository provides developers with a practical introduction to various machine learning approaches for predicting stock market movements. Each method includes:
- Clear explanations of the approach
- Working code examples
- Financial terminology definitions
- Pros and cons for practical application
- When to use each method

**Target Audience**: Developers interested in financial predictive modelling who want to understand the landscape of available approaches and decide which direction to pursue for their own financial modelling projects and applications.

## 🎯 What This Repository Covers

### 1. Traditional Statistical Methods (`src/statistical/`)
Methods that have been used in quantitative finance for decades:
- **ARIMA** (AutoRegressive Integrated Moving Average): Time series forecasting based on historical patterns
- **GARCH** (Generalized AutoRegressive Conditional Heteroskedasticity): Modeling volatility over time
- **Exponential Smoothing**: Weighted averaging with decay for trend analysis

### 2. Classical Machine Learning (`src/classical_ml/`)
Traditional ML algorithms adapted for financial prediction:
- **Random Forest**: Ensemble of decision trees for classification/regression
- **Support Vector Machines (SVM)**: Finding optimal decision boundaries
- **Gradient Boosting (XGBoost, LightGBM)**: Sequential model improvement
- **Linear Models with Regularization**: Ridge and Lasso regression

### 3. Deep Learning Methods (`src/deep_learning/`)
Neural network architectures for sequential data:
- **LSTM** (Long Short-Term Memory): Capturing long-term dependencies in sequences
- **GRU** (Gated Recurrent Unit): Simplified recurrent architecture
- **CNN-LSTM Hybrid**: Combining pattern detection with sequence modeling
- **Temporal Convolutional Networks**: Causal convolutions for time series

### 4. Modern Transformer-Based Approaches (`src/transformers/`)
Attention mechanisms:
- **Temporal Fusion Transformer**: Multi-horizon forecasting with interpretability
- **Time Series Transformer**: Self-attention for temporal patterns
- **Informer**: Efficient long-sequence modeling

### 5. Reinforcement Learning (`src/reinforcement_learning/`)
Learning trading strategies through interaction:
- **Deep Q-Network (DQN)**: Learning optimal actions for trading
- **Policy Gradient Methods**: Direct strategy optimization
- **Actor-Critic**: Combining value and policy learning

### 6. Experimental & Hybrid Methods (`src/experimental/`)
Novel approaches:
- **Graph Neural Networks**: Modeling stock relationships as networks
- **Ensemble Methods**: Combining multiple models
- **Sentiment Analysis Integration**: Using news and social media
- **Meta-Learning**: Learning to adapt quickly to market regimes

## 🏗️ Repository Structure

```
stock_market/
├── README.md                          # This file
├── GLOSSARY.md                        # Financial terminology reference
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
│
├── data/                             # Data storage
│   ├── raw/                          # Downloaded data
│   ├── processed/                    # Preprocessed features
│   └── sample/                       # Sample datasets for examples
│
├── src/                              # Source code
│   ├── statistical/                  # Traditional methods
│   │   ├── arima_model.py
│   │   ├── garch_model.py
│   │   └── exp_smoothing.py
│   │
│   ├── classical_ml/                 # Classical ML
│   │   ├── random_forest.py
│   │   ├── svm_model.py
│   │   ├── gradient_boosting.py
│   │   └── feature_engineering.py
│   │
│   ├── deep_learning/                # Deep learning
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   ├── cnn_lstm.py
│   │   └── tcn_model.py
│   │
│   ├── transformers/                 # Transformer models
│   │   ├── temporal_fusion.py
│   │   ├── time_series_transformer.py
│   │   └── informer.py
│   │
│   ├── reinforcement_learning/       # RL approaches
│   │   ├── dqn_trader.py
│   │   ├── policy_gradient.py
│   │   └── trading_env.py
│   │
│   ├── experimental/                 # Novel approaches
│   │   ├── graph_neural_net.py
│   │   ├── ensemble_methods.py
│   │   └── sentiment_integration.py
│   │
│   └── utils/                        # Utilities
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── evaluation.py
│       └── visualization.py
│
├── notebooks/                        # Jupyter tutorials
│   ├── 01_data_exploration.ipynb
│   ├── 02_statistical_methods.ipynb
│   ├── 03_classical_ml.ipynb
│   ├── 04_deep_learning.ipynb
│   ├── 05_transformers.ipynb
│   ├── 06_reinforcement_learning.ipynb
│   └── 07_model_comparison.ipynb
│
└── tests/                            # Unit tests
    └── ...
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Either a virtual environment (venv) or conda environment (your choice!)

### Installation

**Option 1: Using Python Virtual Environment (Recommended)**

1. **Create and activate a virtual environment**:
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
``` 

**Option 2: Using Conda Environment**

1. **Create and activate conda environment**:
```bash
conda create -n stock_market python=3.12
conda activate stock_market
```

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

> **Note**: We use `pip` for all dependencies (not `conda install`) because many packages like `yfinance` and `stable-baselines3` are more up-to-date on PyPI. Both venv and conda work fine - choose whichever you prefer!

**3. Download sample data** (we'll use yfinance for easy access):
```python
from src.utils.data_loader import download_sample_data
download_sample_data()
```

### Quick Start Example

```python
from src.utils.data_loader import load_stock_data
from src.classical_ml.random_forest import RandomForestPredictor

# Load data
data = load_stock_data('AAPL', start='2020-01-01', end='2023-12-31')

# Train a simple model
model = RandomForestPredictor()
model.train(data)

# Make predictions
predictions = model.predict(data)
```

## 📊 Key Concepts

### Financial Terminology You'll Encounter

- **Stock Price**: The value of a single share of a company
- **Volume**: Number of shares traded during a time period
- **Returns**: Percentage change in price over time
- **Volatility**: How much the price fluctuates (standard deviation of returns)
- **Technical Indicators**: Mathematical calculations based on price/volume (e.g., Moving Averages, RSI, MACD)
- **Candlestick**: Visual representation showing Open, High, Low, Close (OHLC) prices
- **Bull/Bear Market**: Rising/falling market conditions

See [GLOSSARY.md](GLOSSARY.md) for comprehensive definitions.

### Evaluation Metrics

- **RMSE** (Root Mean Square Error): Average prediction error magnitude
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **Directional Accuracy**: Percentage of correct up/down predictions
- **Sharpe Ratio**: Risk-adjusted returns of a trading strategy
- **Maximum Drawdown**: Largest peak-to-trough decline

## 🎓 Recommended Learning Path

1. **Start with Data**: Run `notebooks/01_data_exploration.ipynb` to understand financial data
2. **Traditional Methods**: Learn ARIMA and statistical approaches (easiest to interpret)
3. **Classical ML**: Move to Random Forests and XGBoost (good baseline performance)
4. **Deep Learning**: Try LSTMs when you need to capture complex patterns
5. **Advanced Topics**: Explore Transformers and RL once comfortable with basics

## ⚖️ Method Comparison

| Method | Complexity | Interpretability | Data Required | Best For |
|--------|-----------|------------------|---------------|----------|
| ARIMA | Low | High | Low | Short-term forecasting, understanding trends |
| Random Forest | Medium | Medium | Medium | Feature importance, baseline models |
| LSTM | High | Low | High | Complex sequential patterns |
| Transformers | Very High | Low | Very High | Multi-horizon forecasting, multiple assets |
| Reinforcement Learning | Very High | Low | High | Learning complete trading strategies |

## ⚠️ Important Disclaimers

1. **Past Performance ≠ Future Results**: Historical patterns may not repeat
2. **Market Efficiency**: Many believe markets are too efficient to predict consistently
3. **Overfitting Risk**: Models can learn noise instead of signal
4. **Transaction Costs**: Real trading involves fees that impact profitability
5. **This is Educational**: Not financial advice; always do your own research

## 🔧 Advanced Topics

- **Feature Engineering**: Creating predictive variables from raw data
- **Walk-Forward Analysis**: Realistic backtesting with rolling windows
- **Risk Management**: Position sizing and stop-loss strategies
- **Portfolio Optimization**: Multi-asset allocation
- **Market Regime Detection**: Identifying changing market conditions

## 📚 Additional Resources

- **Books**: "Advances in Financial Machine Learning" by Marcos López de Prado
- **Courses**: Coursera's "Machine Learning for Trading"
- **Papers**: Check `docs/papers.md` for seminal research
- **Datasets**: Yahoo Finance, Alpha Vantage, Quandl

## 📄 License

MIT License - feel free to use this for learning and development.

---
