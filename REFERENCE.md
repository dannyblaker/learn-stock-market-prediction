# 🚀 Quick Reference Card

A one-page reference for common tasks and commands.

## Installation

```bash
conda activate stock_market
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
# Run demo
python demo.py

# Launch Jupyter
jupyter notebook notebooks/01_getting_started.ipynb
```

## Common Code Snippets

### Load Data
```python
from src.utils.data_loader import load_stock_data
data = load_stock_data('AAPL', start='2020-01-01', end='2023-12-31')
```

### Add Features
```python
from src.utils.preprocessing import calculate_technical_indicators
data = calculate_technical_indicators(data)
```

### Train Model
```python
from src.classical_ml.random_forest import RandomForestPredictor
model = RandomForestPredictor(task='classification')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Evaluate
```python
from src.utils.evaluation import evaluate_classification
metrics = evaluate_classification(y_test, predictions)
```

### Visualize
```python
from src.utils.visualization import plot_stock_price, plot_predictions
plot_stock_price(data)
plot_predictions(actual, predicted, dates)
```

## File Locations

| What | Where |
|------|-------|
| Documentation | `README.md`, `GLOSSARY.md`, `QUICKSTART.md` |
| Models | `src/statistical/`, `src/classical_ml/`, `src/deep_learning/` |
| Utilities | `src/utils/` |
| Notebooks | `notebooks/` |
| Demo | `demo.py` |

## Model Quick Access

```python
# ARIMA
from src.statistical.arima_model import ARIMAPredictor
model = ARIMAPredictor(order=(5, 1, 2))

# Random Forest
from src.classical_ml.random_forest import RandomForestPredictor
model = RandomForestPredictor(task='classification')

# LSTM
from src.deep_learning.lstm_model import LSTMPredictor
model = LSTMPredictor(input_size=5, sequence_length=60)

# RL Environment
from src.reinforcement_learning.trading_env import TradingEnv
env = TradingEnv(data, initial_balance=10000)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `pip install -e .` or `export PYTHONPATH="$PWD"` |
| No data | Check internet, try `load_sample_data('AAPL')` |
| Slow training | Reduce data size, use GPU, lower complexity |
| Out of memory | Reduce batch_size, sequence_length |

## Key Metrics

```python
from src.utils.evaluation import (
    evaluate_regression,      # RMSE, MAE, R²
    evaluate_classification,  # Accuracy, Precision, Recall, F1
    calculate_sharpe_ratio,   # Risk-adjusted return
    calculate_max_drawdown,   # Worst loss
    backtest_trading_strategy # Complete backtest
)
```

## Data Preparation

```python
from src.utils.preprocessing import (
    calculate_returns,
    create_lagged_features,
    create_rolling_features,
    prepare_sequences,
    train_test_split_temporal,
    scale_features,
    create_target_variable
)
```

## Financial Terms (Top 10)

1. **Close**: Final price of the day
2. **Volume**: Shares traded
3. **Returns**: % change in price
4. **Volatility**: Price fluctuation
5. **RSI**: Overbought/oversold indicator (0-100)
6. **MACD**: Trend indicator
7. **Bollinger Bands**: Volatility bands
8. **Sharpe Ratio**: Risk-adjusted returns
9. **Drawdown**: Peak-to-trough decline
10. **Backtest**: Test strategy on historical data

See `GLOSSARY.md` for complete definitions.

## Typical Workflow

```python
# 1. Load
data = load_stock_data('AAPL', start='2020-01-01')

# 2. Engineer features
data = calculate_technical_indicators(data)
data = create_lagged_features(data, columns=['Close'], lags=[1,2,3])
data = create_target_variable(data, target_type='direction')
data = data.dropna()

# 3. Split
from src.utils.preprocessing import train_test_split_temporal
train, val, test = train_test_split_temporal(data)

# 4. Prepare X, y
X_train = train[feature_cols]
y_train = train['Target']

# 5. Train
model = RandomForestPredictor(task='classification')
model.fit(X_train, y_train)

# 6. Predict
predictions = model.predict(test[feature_cols])

# 7. Evaluate
evaluate_classification(test['Target'], predictions)
```

## Command Line Quick Tasks

```bash
# Download sample data
python -c "from src.utils.data_loader import download_sample_data; download_sample_data()"

# Run specific model
python src/statistical/arima_model.py
python src/classical_ml/random_forest.py

# Run tests (if implemented)
pytest tests/

# Format code
black src/
```

## Environment Variables

```bash
# Add to .bashrc or .zshrc
export PYTHONPATH="${PYTHONPATH}:/home/d/Documents/stock_market"
```

## Git Workflow (if using)

```bash
# Initialize
git init
git add .
git commit -m "Initial commit: Stock market ML repository"

# Create branch for experiments
git checkout -b experiment/new-feature

# After changes
git add .
git commit -m "Add new feature"
git checkout main
git merge experiment/new-feature
```

## Performance Tips

- Start with small date ranges (1 year)
- Use CPU for learning, GPU for production
- Cache preprocessed data
- Reduce model complexity first, then increase
- Monitor memory usage with `htop` or Task Manager

## Resources

- **Docs**: `README.md`, `GLOSSARY.md`
- **Examples**: Each file's `if __name__ == "__main__"`
- **Interactive**: `notebooks/01_getting_started.ipynb`
- **Quick fixes**: `QUICKSTART.md`, `INSTALL.md`

## Support

- Check docstrings: `help(function_name)`
- Read inline comments
- See example usage in each module
- Consult documentation files

## Remember

⚠️ **Educational only** - Not financial advice!

---

*Keep this card handy for quick reference while working with the repository.*
