"""Utility modules for stock market prediction."""

from .data_loader import (
    load_stock_data,
    load_multiple_stocks,
    download_sample_data,
    load_sample_data,
    get_sp500_tickers
)

from .preprocessing import (
    calculate_returns,
    calculate_technical_indicators,
    create_lagged_features,
    create_rolling_features,
    prepare_sequences,
    train_test_split_temporal,
    scale_features,
    create_target_variable
)

from .evaluation import (
    evaluate_regression,
    evaluate_classification,
    calculate_directional_accuracy,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    backtest_trading_strategy,
    compare_models
)

from .visualization import (
    plot_stock_price,
    plot_candlestick,
    plot_predictions,
    plot_residuals,
    plot_feature_importance,
    plot_correlation_matrix,
    plot_portfolio_performance,
    plot_training_history,
    plot_bollinger_bands
)

__all__ = [
    # Data loading
    'load_stock_data',
    'load_multiple_stocks',
    'download_sample_data',
    'load_sample_data',
    'get_sp500_tickers',
    
    # Preprocessing
    'calculate_returns',
    'calculate_technical_indicators',
    'create_lagged_features',
    'create_rolling_features',
    'prepare_sequences',
    'train_test_split_temporal',
    'scale_features',
    'create_target_variable',
    
    # Evaluation
    'evaluate_regression',
    'evaluate_classification',
    'calculate_directional_accuracy',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'backtest_trading_strategy',
    'compare_models',
    
    # Visualization
    'plot_stock_price',
    'plot_candlestick',
    'plot_predictions',
    'plot_residuals',
    'plot_feature_importance',
    'plot_correlation_matrix',
    'plot_portfolio_performance',
    'plot_training_history',
    'plot_bollinger_bands'
]
