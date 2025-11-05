"""Stock Market ML Package - Machine Learning for Stock Market Prediction."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main utilities
from src.utils import (
    load_stock_data,
    calculate_technical_indicators,
    evaluate_regression,
    evaluate_classification,
    plot_stock_price,
    plot_predictions
)

__all__ = [
    'load_stock_data',
    'calculate_technical_indicators',
    'evaluate_regression',
    'evaluate_classification',
    'plot_stock_price',
    'plot_predictions'
]
