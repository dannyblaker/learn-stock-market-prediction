"""
Data preprocessing utilities for stock market prediction.

This module provides functions for feature engineering, data transformation,
and preparing data for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


def calculate_returns(data: pd.DataFrame, 
                      price_col: str = 'Close',
                      periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Calculate returns over multiple time periods.
    
    Returns represent the percentage change in price, which is often
    more stationary than raw prices and easier for models to learn.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with price columns
    price_col : str
        Column name for price (default: 'Close')
    periods : List[int]
        List of periods to calculate returns for
        
    Returns
    -------
    pd.DataFrame
        Data with return columns added
        
    Examples
    --------
    >>> data = calculate_returns(data, periods=[1, 5, 10])
    >>> # Creates columns: Returns_1, Returns_5, Returns_10
    """
    df = data.copy()
    
    for period in periods:
        # Simple returns
        df[f'Returns_{period}'] = df[price_col].pct_change(periods=period)
        
        # Log returns (preferred for modeling)
        df[f'Log_Returns_{period}'] = np.log(
            df[price_col] / df[price_col].shift(period)
        )
    
    return df


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive technical indicators.
    
    Technical indicators are mathematical calculations based on price and
    volume that traders use to identify trends and potential reversals.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with OHLC and Volume columns
        
    Returns
    -------
    pd.DataFrame
        Data with technical indicator columns
        
    Indicators Added
    ----------------
    - Moving Averages (SMA, EMA)
    - MACD (Moving Average Convergence Divergence)
    - RSI (Relative Strength Index)
    - Bollinger Bands
    - ATR (Average True Range)
    - Volume indicators
    """
    df = data.copy()
    
    # === Moving Averages ===
    for window in [5, 10, 20, 50, 200]:
        # Simple Moving Average
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Exponential Moving Average (gives more weight to recent prices)
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # === MACD (trend-following momentum indicator) ===
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # === RSI (Relative Strength Index - momentum oscillator) ===
    # Measures speed and magnitude of price changes
    # Values: 0-100, >70 = overbought, <30 = oversold
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # === Bollinger Bands (volatility indicator) ===
    # Shows how far price is from its average
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
    
    # === ATR (Average True Range - volatility measure) ===
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # === Volume Indicators ===
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # On-Balance Volume (cumulative volume indicator)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # === Price Momentum ===
    for period in [5, 10, 20]:
        df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        df[f'ROC_{period}'] = (
            (df['Close'] - df['Close'].shift(period)) / 
            df['Close'].shift(period) * 100
        )
    
    # === Volatility ===
    for window in [5, 10, 20]:
        df[f'Volatility_{window}'] = (
            df['Returns_1'].rolling(window=window).std() if 'Returns_1' in df.columns
            else df['Close'].pct_change().rolling(window=window).std()
        )
    
    return df


def create_lagged_features(data: pd.DataFrame, 
                          columns: List[str],
                          lags: List[int]) -> pd.DataFrame:
    """
    Create lagged versions of features.
    
    Lagged features are previous values of a variable, allowing models
    to learn from historical patterns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    columns : List[str]
        Columns to create lags for
    lags : List[int]
        List of lag periods (e.g., [1, 2, 3, 5, 10])
        
    Returns
    -------
    pd.DataFrame
        Data with lagged feature columns
        
    Examples
    --------
    >>> data = create_lagged_features(
    ...     data, 
    ...     columns=['Close', 'Volume'],
    ...     lags=[1, 2, 3, 5]
    ... )
    >>> # Creates: Close_lag_1, Close_lag_2, ..., Volume_lag_5
    """
    df = data.copy()
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping...")
            continue
            
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df


def create_rolling_features(data: pd.DataFrame,
                           columns: List[str],
                           windows: List[int],
                           functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    Rolling features capture recent trends and patterns over specific
    time windows, helping models understand short-term dynamics.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    columns : List[str]
        Columns to compute rolling statistics for
    windows : List[int]
        Window sizes (e.g., [5, 10, 20])
    functions : List[str]
        Statistics to compute: 'mean', 'std', 'min', 'max', 'median'
        
    Returns
    -------
    pd.DataFrame
        Data with rolling feature columns
    """
    df = data.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for window in windows:
            if 'mean' in functions:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
            if 'std' in functions:
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
            if 'min' in functions:
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window).min()
            if 'max' in functions:
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window).max()
            if 'median' in functions:
                df[f'{col}_roll_median_{window}'] = df[col].rolling(window).median()
    
    return df


def prepare_sequences(data: np.ndarray, 
                     sequence_length: int,
                     target_column: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequential data for deep learning models (LSTM, GRU, etc.).
    
    Converts time series into sequences of fixed length that can be
    used for training recurrent neural networks.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of shape (samples, features)
    sequence_length : int
        Number of time steps in each sequence
    target_column : int
        Column index to use as target (default: 0)
        
    Returns
    -------
    X : np.ndarray
        3D array of shape (samples, sequence_length, features)
    y : np.ndarray
        1D array of targets
        
    Examples
    --------
    >>> # Create sequences of 60 days to predict next day
    >>> X, y = prepare_sequences(data, sequence_length=60)
    >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        # Get sequence of specified length
        X.append(data[i:i + sequence_length])
        # Get target (next value after sequence)
        y.append(data[i + sequence_length, target_column])
    
    return np.array(X), np.array(y)


def train_test_split_temporal(data: pd.DataFrame,
                              test_size: float = 0.2,
                              validation_size: float = 0.1) -> Tuple[pd.DataFrame, ...]:
    """
    Split time series data into train, validation, and test sets.
    
    IMPORTANT: Unlike random splitting, this preserves temporal order.
    Train comes first, then validation, then test - as would happen in reality.
    This prevents look-ahead bias.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    test_size : float
        Proportion of data for testing (default: 0.2 = 20%)
    validation_size : float
        Proportion of data for validation (default: 0.1 = 10%)
        
    Returns
    -------
    train_data, val_data, test_data : pd.DataFrame
        Split datasets in temporal order
        
    Examples
    --------
    >>> train, val, test = train_test_split_temporal(data, test_size=0.2, validation_size=0.1)
    >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    n = len(data)
    
    # Calculate split points
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - validation_size))
    
    # Split data
    train_data = data.iloc[:val_start]
    val_data = data.iloc[val_start:test_start]
    test_data = data.iloc[test_start:]
    
    print(f"Data split:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/n*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/n*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/n*100:.1f}%)")
    
    return train_data, val_data, test_data


def scale_features(train_data: pd.DataFrame,
                  val_data: pd.DataFrame,
                  test_data: pd.DataFrame,
                  method: str = 'standard',
                  exclude_columns: List[str] = None) -> Tuple[pd.DataFrame, ...]:
    """
    Scale features using training data statistics.
    
    Scaling is important for many ML algorithms. We fit the scaler only
    on training data to prevent data leakage.
    
    Parameters
    ----------
    train_data, val_data, test_data : pd.DataFrame
        Split datasets
    method : str
        'standard' (mean=0, std=1) or 'minmax' (range 0-1)
    exclude_columns : List[str], optional
        Columns not to scale (e.g., target variable)
        
    Returns
    -------
    train_scaled, val_scaled, test_scaled : pd.DataFrame
        Scaled datasets
    scaler : StandardScaler or MinMaxScaler
        Fitted scaler object (for inverse transform later)
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Select columns to scale
    cols_to_scale = [col for col in train_data.columns if col not in exclude_columns]
    
    # Initialize scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit on training data only
    scaler.fit(train_data[cols_to_scale])
    
    # Transform all sets
    train_scaled = train_data.copy()
    val_scaled = val_data.copy()
    test_scaled = test_data.copy()
    
    train_scaled[cols_to_scale] = scaler.transform(train_data[cols_to_scale])
    val_scaled[cols_to_scale] = scaler.transform(val_data[cols_to_scale])
    test_scaled[cols_to_scale] = scaler.transform(test_data[cols_to_scale])
    
    return train_scaled, val_scaled, test_scaled, scaler


def create_target_variable(data: pd.DataFrame,
                          target_type: str = 'price',
                          horizon: int = 1,
                          price_col: str = 'Close') -> pd.DataFrame:
    """
    Create target variable for prediction.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data
    target_type : str
        Type of target:
        - 'price': Future price value
        - 'return': Future return value
        - 'direction': Binary up/down classification (1/0)
        - 'change': Future price change amount
    horizon : int
        How many periods ahead to predict (default: 1 day)
    price_col : str
        Price column to base target on
        
    Returns
    -------
    pd.DataFrame
        Data with 'Target' column added
    """
    df = data.copy()
    
    if target_type == 'price':
        # Predict future price
        df['Target'] = df[price_col].shift(-horizon)
        
    elif target_type == 'return':
        # Predict future return
        df['Target'] = df[price_col].pct_change(periods=horizon).shift(-horizon)
        
    elif target_type == 'direction':
        # Predict whether price will go up (1) or down (0)
        future_price = df[price_col].shift(-horizon)
        df['Target'] = (future_price > df[price_col]).astype(int)
        
    elif target_type == 'change':
        # Predict price change amount
        df['Target'] = (df[price_col].shift(-horizon) - df[price_col])
        
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    return df


if __name__ == "__main__":
    # Example usage
    from data_loader import load_stock_data
    
    print("Stock Market Data Preprocessing")
    print("=" * 50)
    
    # Load sample data
    data = load_stock_data('AAPL', start='2020-01-01')
    print(f"\nOriginal data shape: {data.shape}")
    
    # Calculate returns
    data = calculate_returns(data)
    print(f"After returns: {data.shape}")
    
    # Add technical indicators
    data = calculate_technical_indicators(data)
    print(f"After technical indicators: {data.shape}")
    
    # Create lagged features
    data = create_lagged_features(data, columns=['Close', 'Volume'], lags=[1, 2, 3, 5])
    print(f"After lagged features: {data.shape}")
    
    # Create target
    data = create_target_variable(data, target_type='direction', horizon=1)
    
    # Remove NaN values
    data = data.dropna()
    print(f"After removing NaN: {data.shape}")
    
    print("\nSample of processed data:")
    print(data.tail())
