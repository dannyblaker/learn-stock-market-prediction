"""
Model evaluation utilities for stock market prediction.

This module provides metrics and functions to evaluate the performance
of various prediction models, including both regression and classification metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


def evaluate_regression(y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate regression predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    verbose : bool
        Whether to print results
        
    Returns
    -------
    dict
        Dictionary of metric names and values
        
    Metrics
    -------
    - RMSE: Root Mean Squared Error (lower is better)
    - MAE: Mean Absolute Error (lower is better)
    - R²: R-squared coefficient of determination (higher is better, max 1.0)
    - MAPE: Mean Absolute Percentage Error (lower is better)
    """
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape = np.where(np.isfinite(mape), mape, 0)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': float(np.mean(mape))
    }
    
    if verbose:
        print("Regression Metrics:")
        print("-" * 40)
        print(f"RMSE:  {rmse:.4f}")
        print(f"MAE:   {mae:.4f}")
        print(f"R²:    {r2:.4f}")
        print(f"MAPE:  {metrics['MAPE']:.2f}%")
    
    return metrics


def evaluate_classification(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate classification predictions (e.g., up/down direction).
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual class labels
    y_pred : np.ndarray
        Predicted class labels
    verbose : bool
        Whether to print results
        
    Returns
    -------
    dict
        Dictionary of metric names and values
        
    Metrics
    -------
    - Accuracy: Percentage of correct predictions
    - Precision: True positives / (True positives + False positives)
    - Recall: True positives / (True positives + False negatives)
    - F1: Harmonic mean of precision and recall
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    
    if verbose:
        print("Classification Metrics:")
        print("-" * 40)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))
    
    return metrics


def calculate_directional_accuracy(y_true: np.ndarray,
                                   y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy for price predictions.
    
    Even if predicted prices are not exact, getting the direction right
    (up vs down) is valuable for trading decisions.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual prices
    y_pred : np.ndarray
        Predicted prices
        
    Returns
    -------
    float
        Directional accuracy (0 to 1)
        
    Examples
    --------
    >>> # If actual goes up and prediction goes up = correct
    >>> # If actual goes down and prediction goes down = correct
    >>> da = calculate_directional_accuracy(actual_prices, predicted_prices)
    """
    # Calculate actual direction (1 if up, 0 if down)
    actual_direction = np.sign(np.diff(y_true))
    predicted_direction = np.sign(np.diff(y_pred))
    
    # Calculate accuracy
    correct = (actual_direction == predicted_direction).sum()
    total = len(actual_direction)
    
    return correct / total


def calculate_sharpe_ratio(returns: np.ndarray,
                          risk_free_rate: float = 0.02,
                          periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    The Sharpe Ratio measures risk-adjusted returns. It tells you how much
    return you're getting per unit of risk (volatility) taken.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of period returns (e.g., daily returns)
    risk_free_rate : float
        Annual risk-free rate (default: 0.02 = 2%)
    periods_per_year : int
        Number of trading periods per year (252 for daily, 12 for monthly)
        
    Returns
    -------
    float
        Annualized Sharpe Ratio
        
    Interpretation
    --------------
    - < 1.0: Poor risk-adjusted return
    - 1.0-2.0: Good
    - 2.0-3.0: Very good
    - > 3.0: Excellent (rare)
    
    Examples
    --------
    >>> daily_returns = portfolio_values.pct_change().dropna()
    >>> sharpe = calculate_sharpe_ratio(daily_returns, periods_per_year=252)
    """
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # Calculate Sharpe Ratio
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    
    # Annualize
    sharpe_annualized = sharpe * np.sqrt(periods_per_year)
    
    return sharpe_annualized


def calculate_max_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from returns.
    
    Maximum drawdown is the largest peak-to-trough decline. It shows
    the worst-case loss an investor would have experienced.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of period returns
        
    Returns
    -------
    max_drawdown : float
        Maximum drawdown as a decimal (e.g., -0.25 = -25%)
    peak_idx : int
        Index of the peak before drawdown
    trough_idx : int
        Index of the trough (lowest point)
        
    Examples
    --------
    >>> returns = [0.01, 0.02, -0.03, -0.05, 0.01, 0.04]
    >>> mdd, peak, trough = calculate_max_drawdown(returns)
    >>> print(f"Max drawdown: {mdd*100:.2f}%")
    """
    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown = np.min(drawdown)
    trough_idx = np.argmin(drawdown)
    
    # Find the peak before the trough
    peak_idx = np.argmax(cumulative[:trough_idx+1]) if trough_idx > 0 else 0
    
    return max_drawdown, peak_idx, trough_idx


def calculate_calmar_ratio(returns: np.ndarray,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Calmar Ratio (return / max drawdown).
    
    The Calmar Ratio measures return relative to maximum drawdown risk.
    Higher values indicate better risk-adjusted performance.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of period returns
    periods_per_year : int
        Number of trading periods per year
        
    Returns
    -------
    float
        Calmar Ratio
    """
    # Annualized return
    total_return = np.prod(1 + returns) - 1
    years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Max drawdown
    max_dd, _, _ = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return np.inf
    
    return annualized_return / abs(max_dd)


def backtest_trading_strategy(predictions: np.ndarray,
                              actual_prices: np.ndarray,
                              initial_capital: float = 10000.0,
                              transaction_cost: float = 0.001) -> Dict:
    """
    Backtest a simple long/short trading strategy.
    
    Strategy: Buy when prediction suggests price will rise,
              sell when prediction suggests price will fall.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted prices or directions
    actual_prices : np.ndarray
        Actual prices
    initial_capital : float
        Starting capital
    transaction_cost : float
        Transaction cost as fraction (0.001 = 0.1%)
        
    Returns
    -------
    dict
        Backtest results including returns, Sharpe ratio, etc.
        
    Note
    ----
    This is a simplified backtest. Real trading involves:
    - Bid-ask spreads
    - Slippage (execution at worse prices)
    - Market impact (your orders affecting prices)
    - Opportunity costs
    """
    # Calculate predicted direction
    pred_direction = np.diff(predictions)
    pred_direction = np.sign(pred_direction)
    
    # Calculate actual returns
    actual_returns = np.diff(actual_prices) / actual_prices[:-1]
    
    # Strategy returns (aligned with predictions)
    # If we predict up (1), we go long and earn actual return
    # If we predict down (-1), we go short and earn -actual return
    strategy_returns = pred_direction * actual_returns[1:]
    
    # Apply transaction costs (every trade)
    # Count position changes
    position_changes = np.diff(np.concatenate([[0], pred_direction]))
    trades = np.abs(position_changes) > 0
    strategy_returns[trades[1:]] -= transaction_cost
    
    # Calculate portfolio value over time
    portfolio_values = initial_capital * np.cumprod(1 + strategy_returns)
    
    # Calculate metrics
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    sharpe = calculate_sharpe_ratio(strategy_returns)
    max_dd, _, _ = calculate_max_drawdown(strategy_returns)
    
    # Win rate
    winning_trades = (strategy_returns > 0).sum()
    total_trades = len(strategy_returns)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    results = {
        'initial_capital': initial_capital,
        'final_capital': portfolio_values[-1],
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd * 100,
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'total_trades': total_trades,
        'portfolio_values': portfolio_values,
        'strategy_returns': strategy_returns
    }
    
    return results


def print_backtest_results(results: Dict):
    """
    Print formatted backtest results.
    
    Parameters
    ----------
    results : dict
        Results from backtest_trading_strategy()
    """
    print("Backtest Results:")
    print("=" * 50)
    print(f"Initial Capital:    ${results['initial_capital']:,.2f}")
    print(f"Final Capital:      ${results['final_capital']:,.2f}")
    print(f"Total Return:       {results['total_return_pct']:,.2f}%")
    print(f"Sharpe Ratio:       {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:       {results['max_drawdown_pct']:.2f}%")
    print(f"Win Rate:           {results['win_rate_pct']:.2f}%")
    print(f"Total Trades:       {results['total_trades']}")
    print("=" * 50)


def compare_models(models_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple models' performance.
    
    Parameters
    ----------
    models_results : dict
        Dictionary mapping model names to their metric dictionaries
        
    Returns
    -------
    pd.DataFrame
        Comparison table of all models
        
    Examples
    --------
    >>> results = {
    ...     'ARIMA': {'RMSE': 2.5, 'MAE': 1.8, 'R2': 0.65},
    ...     'LSTM': {'RMSE': 1.9, 'MAE': 1.4, 'R2': 0.78},
    ...     'XGBoost': {'RMSE': 2.1, 'MAE': 1.6, 'R2': 0.72}
    ... }
    >>> comparison = compare_models(results)
    >>> print(comparison)
    """
    df = pd.DataFrame(models_results).T
    
    # Sort by R² (descending) or RMSE (ascending) if available
    if 'R2' in df.columns:
        df = df.sort_values('R2', ascending=False)
    elif 'RMSE' in df.columns:
        df = df.sort_values('RMSE', ascending=True)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Utilities")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randn(100).cumsum() + 100
    y_pred = y_true + np.random.randn(100) * 2
    
    # Regression evaluation
    print("\nRegression Evaluation:")
    metrics = evaluate_regression(y_true, y_pred)
    
    # Classification evaluation
    print("\n" + "=" * 50)
    print("\nClassification Evaluation:")
    y_true_class = (np.diff(y_true) > 0).astype(int)
    y_pred_class = (np.diff(y_pred) > 0).astype(int)
    class_metrics = evaluate_classification(y_true_class, y_pred_class)
    
    # Calculate directional accuracy
    print("\n" + "=" * 50)
    da = calculate_directional_accuracy(y_true, y_pred)
    print(f"\nDirectional Accuracy: {da:.4f} ({da*100:.2f}%)")
    
    # Backtest
    print("\n" + "=" * 50)
    backtest_results = backtest_trading_strategy(y_pred, y_true)
    print_backtest_results(backtest_results)
