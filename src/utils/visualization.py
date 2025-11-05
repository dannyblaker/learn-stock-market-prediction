"""
Visualization utilities for stock market data and predictions.

This module provides functions to create various charts and plots
for analyzing stock data and model predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_stock_price(data: pd.DataFrame,
                    title: str = "Stock Price",
                    figsize: Tuple[int, int] = (14, 7)):
    """
    Plot stock price over time with volume.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with Close and Volume columns
    title : str
        Chart title
    figsize : tuple
        Figure size (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                     height_ratios=[3, 1], sharex=True)
    
    # Price plot
    ax1.plot(data.index, data['Close'], linewidth=1.5, label='Close Price')
    if 'SMA_20' in data.columns:
        ax1.plot(data.index, data['SMA_20'], linewidth=1, 
                label='20-day SMA', alpha=0.7)
    if 'SMA_50' in data.columns:
        ax1.plot(data.index, data['SMA_50'], linewidth=1, 
                label='50-day SMA', alpha=0.7)
    
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Volume plot
    ax2.bar(data.index, data['Volume'], width=1, alpha=0.7, color='steelblue')
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_candlestick(data: pd.DataFrame,
                    n_days: int = 100,
                    title: str = "Candlestick Chart"):
    """
    Plot candlestick chart (simplified version).
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with OHLC columns
    n_days : int
        Number of recent days to plot
    title : str
        Chart title
        
    Note
    ----
    For interactive candlestick charts, consider using plotly:
    import plotly.graph_objects as go
    """
    data = data.tail(n_days).copy()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Determine color for each day
    colors = ['green' if close >= open_ else 'red' 
              for open_, close in zip(data['Open'], data['Close'])]
    
    # Plot high-low lines
    for i, (idx, row) in enumerate(data.iterrows()):
        ax.plot([i, i], [row['Low'], row['High']], 
               color='black', linewidth=0.5)
        
        # Plot open-close box
        height = abs(row['Close'] - row['Open'])
        bottom = min(row['Open'], row['Close'])
        ax.bar(i, height, width=0.6, bottom=bottom, 
              color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Set x-axis labels
    step = max(1, len(data) // 10)
    ax.set_xticks(range(0, len(data), step))
    ax.set_xticklabels([data.index[i].strftime('%Y-%m-%d') 
                        for i in range(0, len(data), step)], 
                       rotation=45)
    
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_predictions(actual: np.ndarray,
                    predicted: np.ndarray,
                    dates: Optional[pd.DatetimeIndex] = None,
                    title: str = "Actual vs Predicted",
                    figsize: Tuple[int, int] = (14, 7)):
    """
    Plot actual vs predicted values.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values
    dates : pd.DatetimeIndex, optional
        Date index for x-axis
    title : str
        Chart title
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    x = dates if dates is not None else range(len(actual))
    
    # Plot actual vs predicted
    ax1.plot(x, actual, label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(x, predicted, label='Predicted', linewidth=2, alpha=0.8)
    ax1.fill_between(x, actual, predicted, alpha=0.2)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot prediction errors
    errors = actual - predicted
    ax2.bar(x, errors, alpha=0.6, color='coral')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    if dates is not None:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_residuals(actual: np.ndarray,
                  predicted: np.ndarray,
                  title: str = "Residual Analysis"):
    """
    Plot residual analysis for regression models.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values
    title : str
        Chart title
    """
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals vs Predicted
    axes[0, 0].scatter(predicted, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals over time
    axes[1, 1].plot(residuals, linewidth=1)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names: List[str],
                           importances: np.ndarray,
                           top_n: int = 20,
                           title: str = "Feature Importance"):
    """
    Plot feature importance for tree-based models.
    
    Parameters
    ----------
    feature_names : list
        Names of features
    importances : np.ndarray
        Importance scores
    top_n : int
        Number of top features to show
    title : str
        Chart title
    """
    # Sort by importance
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], alpha=0.8)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(data: pd.DataFrame,
                           figsize: Tuple[int, int] = (12, 10),
                           title: str = "Correlation Matrix"):
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with features to correlate
    figsize : tuple
        Figure size
    title : str
        Chart title
    """
    corr = data.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_portfolio_performance(portfolio_values: np.ndarray,
                              dates: Optional[pd.DatetimeIndex] = None,
                              benchmark: Optional[np.ndarray] = None,
                              title: str = "Portfolio Performance"):
    """
    Plot portfolio value over time.
    
    Parameters
    ----------
    portfolio_values : np.ndarray
        Portfolio values over time
    dates : pd.DatetimeIndex, optional
        Date index
    benchmark : np.ndarray, optional
        Benchmark values (e.g., S&P 500)
    title : str
        Chart title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    
    x = dates if dates is not None else range(len(portfolio_values))
    
    # Portfolio value
    ax1.plot(x, portfolio_values, label='Portfolio', linewidth=2)
    if benchmark is not None:
        ax1.plot(x, benchmark, label='Benchmark', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1] * 100
    x_returns = x[1:] if dates is not None else range(len(returns))
    ax2.bar(x_returns, returns, alpha=0.6, 
           color=['green' if r >= 0 else 'red' for r in returns])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Returns (%)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    if dates is not None:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict,
                         metrics: List[str] = ['loss'],
                         title: str = "Training History"):
    """
    Plot training history for neural networks.
    
    Parameters
    ----------
    history : dict
        Training history with metric values per epoch
        Keys should be metric names, values should be lists
    metrics : list
        Metrics to plot
    title : str
        Chart title
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in history:
            axes[i].plot(history[metric], label=f'Train {metric}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                axes[i].plot(history[val_metric], label=f'Val {metric}')
            
            axes[i].set_xlabel('Epoch', fontsize=12)
            axes[i].set_ylabel(metric.capitalize(), fontsize=12)
            axes[i].set_title(f'{metric.capitalize()} over Epochs')
            axes[i].legend(loc='best')
            axes[i].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_bollinger_bands(data: pd.DataFrame,
                        n_days: int = 100,
                        title: str = "Bollinger Bands"):
    """
    Plot Bollinger Bands.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with Close, BB_Upper, BB_Lower columns
    n_days : int
        Number of recent days to plot
    title : str
        Chart title
    """
    data = data.tail(n_days)
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(data.index, data['Close'], label='Close Price', linewidth=2)
    
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        plt.plot(data.index, data['BB_Upper'], 'r--', label='Upper Band', alpha=0.7)
        plt.plot(data.index, data['BB_Lower'], 'g--', label='Lower Band', alpha=0.7)
        plt.fill_between(data.index, data['BB_Lower'], data['BB_Upper'], 
                        alpha=0.1, color='gray')
    
    if 'SMA_20' in data.columns:
        plt.plot(data.index, data['SMA_20'], 'b-', label='20-day SMA', alpha=0.7)
    
    plt.ylabel('Price ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization Utilities")
    print("=" * 50)
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    price = 100 + np.random.randn(n).cumsum()
    volume = np.random.randint(1000000, 5000000, n)
    
    data = pd.DataFrame({
        'Close': price,
        'Open': price + np.random.randn(n) * 2,
        'High': price + np.abs(np.random.randn(n) * 3),
        'Low': price - np.abs(np.random.randn(n) * 3),
        'Volume': volume,
        'SMA_20': pd.Series(price).rolling(20).mean(),
        'SMA_50': pd.Series(price).rolling(50).mean()
    }, index=dates)
    
    # Plot stock price
    plot_stock_price(data, title="Sample Stock Price")
    
    # Generate predictions
    predicted = price + np.random.randn(n) * 5
    
    # Plot predictions
    plot_predictions(price, predicted, dates, title="Sample Predictions")
