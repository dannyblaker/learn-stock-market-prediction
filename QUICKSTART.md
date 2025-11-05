# Quick Start Guide

This guide will get you up and running with the stock market prediction repository in minutes.

## Installation

### 1. Set Up Python Environment

**Option A: Python Virtual Environment** (recommended for most users)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Option B: Conda Environment** (if you prefer conda)
```bash
conda create -n stock_market python=3.10
conda activate stock_market
```

> Choose either option - both work perfectly! We use `pip` for all packages.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- Data science libraries (numpy, pandas, scipy)
- Machine learning frameworks (scikit-learn, xgboost, torch)
- Financial data APIs (yfinance)
- Visualization tools (matplotlib, seaborn, plotly)

### 3. Download Sample Data

```python
from src.utils.data_loader import download_sample_data
download_sample_data()
```

Or run from terminal:
```bash
python -c "from src.utils.data_loader import download_sample_data; download_sample_data()"
```

## Your First Prediction

### Option 1: Run the Getting Started Notebook (Recommended)

```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

The notebook automatically configures the Python path for you!

### Option 2: Quick Python Script

If running Python scripts directly, make sure you're in the project root directory:

```python
from src.utils.data_loader import load_stock_data
from src.utils.preprocessing import calculate_technical_indicators, create_target_variable
from src.classical_ml.random_forest import RandomForestPredictor

# Load data
data = load_stock_data('AAPL', start='2020-01-01', end='2023-12-31')

# Add features
data = calculate_technical_indicators(data)
data = create_target_variable(data, target_type='direction', horizon=1)
data = data.dropna()

# Prepare data
feature_cols = [col for col in data.columns if col != 'Target']
X = data[feature_cols]
y = data['Target']

# Split (80% train, 20% test)
split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
model = RandomForestPredictor(task='classification')
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
from src.utils.evaluation import evaluate_classification
evaluate_classification(y_test.values, predictions)
```

## Exploring Different Approaches

### Statistical Methods

```python
from src.statistical.arima_model import ARIMAPredictor

model = ARIMAPredictor(order=(5, 1, 2))
model.fit(data['Close'])
forecast = model.predict(steps=30)
```

### Deep Learning

```python
from src.deep_learning.lstm_model import LSTMPredictor

model = LSTMPredictor(input_size=5, sequence_length=60)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Reinforcement Learning

```python
from src.reinforcement_learning.trading_env import TradingEnv

env = TradingEnv(data, initial_balance=10000)
# Train your RL agent here
```

## Common Tasks

### Load Different Stocks

```python
# Single stock
apple = load_stock_data('AAPL', start='2020-01-01')

# Multiple stocks
from src.utils.data_loader import load_multiple_stocks
stocks = load_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'])
```

### Create Custom Features

```python
from src.utils.preprocessing import (
    calculate_returns,
    create_lagged_features,
    create_rolling_features
)

# Add returns
data = calculate_returns(data, periods=[1, 5, 10, 20])

# Add lagged features
data = create_lagged_features(data, columns=['Close', 'Volume'], lags=[1, 2, 3, 5])

# Add rolling statistics
data = create_rolling_features(data, columns=['Close'], windows=[10, 20, 50])
```

### Visualize Results

```python
from src.utils.visualization import (
    plot_stock_price,
    plot_predictions,
    plot_feature_importance
)

plot_stock_price(data, title='Stock Analysis')
plot_predictions(actual, predicted, dates=data.index)
plot_feature_importance(feature_names, importances)
```

### Backtest Trading Strategy

```python
from src.utils.evaluation import backtest_trading_strategy, print_backtest_results

results = backtest_trading_strategy(
    predictions=predicted_prices,
    actual_prices=actual_prices,
    initial_capital=10000,
    transaction_cost=0.001
)

print_backtest_results(results)
```

## Next Steps

1. **Read the GLOSSARY.md** - Understand financial terminology
2. **Explore notebooks/** - Step-by-step tutorials for each approach
3. **Check src/** - Dive into implementation details
4. **Experiment** - Try different stocks, features, and models

## Troubleshooting

### Import Errors

Make sure you're in the project root directory:
```bash
cd /path/to/stock_market
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Or install as package:
```bash
pip install -e .
```

### Data Download Issues

If yfinance fails:
- Check internet connection
- Try different date ranges
- Use sample data: `load_sample_data('AAPL')`

### Memory Issues

For large datasets:
- Reduce date range
- Use fewer features
- Process in batches
- Use generators for sequences

### GPU Not Detected (PyTorch)

Check CUDA availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

If False, PyTorch will use CPU (slower but works fine for learning).

## Getting Help

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check docstrings and comments in code
- **Examples**: See `if __name__ == "__main__":` blocks in each module

## Important Reminders

⚠️ **This is for education only**
- Not financial advice
- Past performance ≠ future results
- Always validate thoroughly
- Consider transaction costs
- Understand the risks

Happy learning! 🚀📈
