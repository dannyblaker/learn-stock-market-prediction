#!/usr/bin/env python3
"""
Demo script showcasing different ML approaches for stock market prediction.

This script demonstrates the main approaches covered in the repository:
1. Data loading and exploration
2. Feature engineering
3. Traditional statistical methods (ARIMA)
4. Classical ML (Random Forest)
5. Evaluation and comparison

Run: python demo.py
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print(" Stock Market Prediction with Machine Learning - Demo")
print("=" * 70)
print("\nThis demo will walk through different approaches to predict stock prices.")
print("We'll use Apple (AAPL) stock data from 2020-2023.\n")

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: Loading Data")
print("=" * 70)

from src.utils.data_loader import load_stock_data

try:
    data = load_stock_data('AAPL', start='2020-01-01', end='2023-12-31')
    print(f"✓ Successfully loaded {len(data)} days of AAPL data")
    print(f"\nPrice range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"Average daily volume: {data['Volume'].mean()/1e6:.1f}M shares")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    print("\nPlease check your internet connection and try again.")
    sys.exit(1)

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Feature Engineering")
print("=" * 70)

from src.utils.preprocessing import (
    calculate_technical_indicators,
    create_lagged_features,
    create_target_variable,
    train_test_split_temporal
)

print("\nAdding technical indicators (RSI, MACD, Bollinger Bands, etc.)...")
data = calculate_technical_indicators(data)

print("Adding lagged features (yesterday's values)...")
data = create_lagged_features(
    data,
    columns=['Close', 'Volume', 'RSI'],
    lags=[1, 2, 3, 5]
)

print("Creating target variable (will price go up tomorrow?)...")
data = create_target_variable(data, target_type='direction', horizon=1)

# Clean data
data = data.dropna()
print(f"\n✓ Feature engineering complete")
print(f"  Total features: {data.shape[1] - 1}")  # -1 for target
print(f"  Clean samples: {len(data)}")

# =============================================================================
# 3. DATA SPLITTING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Splitting Data")
print("=" * 70)

# Prepare features and target
feature_cols = [col for col in data.columns if col != 'Target']
X = data[feature_cols]
y = data['Target']

# Split temporally (preserving time order)
train_data, val_data, test_data = train_test_split_temporal(
    data, test_size=0.2, validation_size=0.1
)

X_train = train_data[feature_cols]
y_train = train_data['Target']
X_test = test_data[feature_cols]
y_test = test_data['Target']

up_pct = y_train.mean() * 100
print(f"\n✓ Data split complete")
print(f"  Training set: up days = {up_pct:.1f}%")

# =============================================================================
# 4. BASELINE: RANDOM FOREST
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Training Random Forest (Classical ML)")
print("=" * 70)

from src.classical_ml.random_forest import RandomForestPredictor
from src.utils.evaluation import evaluate_classification

print("\nTraining Random Forest classifier...")
rf_model = RandomForestPredictor(
    task='classification',
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("\nMaking predictions...")
rf_predictions = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
rf_metrics = evaluate_classification(y_test.values, rf_predictions, verbose=True)

print("\nTop 5 Most Important Features:")
importance_df = rf_model.get_feature_importance(top_n=5)
for idx, row in importance_df.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# =============================================================================
# 5. COMPARISON WITH SIMPLE BASELINE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Comparison with Baseline")
print("=" * 70)

import numpy as np

# Simple baseline: predict most common class
baseline_pred = np.full_like(rf_predictions, y_train.mode()[0])
baseline_acc = (baseline_pred == y_test.values).mean()

print(f"\nBaseline (predict most common class): {baseline_acc:.4f}")
print(f"Random Forest:                         {rf_metrics['Accuracy']:.4f}")
print(f"Improvement:                           +{(rf_metrics['Accuracy'] - baseline_acc)*100:.2f}%")

# =============================================================================
# 6. PRACTICAL INSIGHTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Practical Insights")
print("=" * 70)

from src.utils.evaluation import backtest_trading_strategy

# Convert predictions to prices (simple strategy)
test_prices = test_data['Close'].values

# Simple backtest
print("\nBacktesting trading strategy...")
print("(Strategy: Buy when predicted up, Sell when predicted down)")

try:
    # Create simple price predictions based on direction
    predicted_prices = test_prices.copy()
    for i in range(1, len(predicted_prices)):
        if rf_predictions[i-1] == 1:  # Predicted up
            predicted_prices[i] = test_prices[i-1] * 1.01  # Predict 1% increase
        else:
            predicted_prices[i] = test_prices[i-1] * 0.99  # Predict 1% decrease
    
    backtest_results = backtest_trading_strategy(
        predictions=predicted_prices,
        actual_prices=test_prices,
        initial_capital=10000,
        transaction_cost=0.001
    )
    
    from src.utils.evaluation import print_backtest_results
    print_backtest_results(backtest_results)
    
except Exception as e:
    print(f"Note: Backtest skipped ({e})")

# =============================================================================
# 7. NEXT STEPS
# =============================================================================
print("\n" + "=" * 70)
print("Next Steps & Resources")
print("=" * 70)

print("""
✓ You've successfully run the demo!

To explore further:

1. Try different approaches:
   - Statistical: src/statistical/arima_model.py
   - Deep Learning: src/deep_learning/lstm_model.py
   - RL: src/reinforcement_learning/trading_env.py

2. Explore notebooks:
   - notebooks/01_getting_started.ipynb
   - Complete step-by-step tutorials

3. Read documentation:
   - GLOSSARY.md - Financial terminology
   - QUICKSTART.md - Quick reference
   - README.md - Full overview

4. Experiment with:
   - Different stocks (change ticker symbol)
   - Different time periods
   - Different features
   - Different models

5. Learn more:
   - Check feature importance to understand what matters
   - Try hyperparameter tuning
   - Combine multiple models (ensemble)
   - Add more sophisticated features

Important Reminders:
⚠️  This is educational software only
⚠️  Not financial advice
⚠️  Past performance ≠ future results
⚠️  Always validate thoroughly before real-world use

Happy learning! 🚀📈
""")

print("=" * 70)
print("Demo Complete!")
print("=" * 70)
