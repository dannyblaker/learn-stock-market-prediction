"""
Data loading utilities for stock market data.

This module provides functions to download and load stock market data
from various sources, primarily using yfinance for easy access.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def load_stock_data(ticker: str, 
                    start: str = None, 
                    end: str = None, 
                    interval: str = '1d') -> pd.DataFrame:
    """
    Load stock data for a given ticker symbol.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    start : str, optional
        Start date in 'YYYY-MM-DD' format. Defaults to 5 years ago.
    end : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today.
    interval : str, optional
        Data interval: '1d', '1wk', '1mo', etc. Defaults to '1d'.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        Index: DatetimeIndex
        
    Examples
    --------
    >>> data = load_stock_data('AAPL', start='2020-01-01', end='2023-12-31')
    >>> print(data.head())
    """
    # Set default dates if not provided
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    if start is None:
        start = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    print(f"Downloading {ticker} data from {start} to {end}...")
    
    try:
        # Download data using yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Remove timezone info for easier processing
        data.index = data.index.tz_localize(None)
        
        print(f"Successfully loaded {len(data)} rows of data")
        return data
    
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise


def load_multiple_stocks(tickers: list, 
                         start: str = None, 
                         end: str = None) -> dict:
    """
    Load data for multiple stock tickers.
    
    Parameters
    ----------
    tickers : list
        List of ticker symbols
    start : str, optional
        Start date in 'YYYY-MM-DD' format
    end : str, optional
        End date in 'YYYY-MM-DD' format
        
    Returns
    -------
    dict
        Dictionary mapping ticker -> DataFrame
        
    Examples
    --------
    >>> data = load_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'])
    >>> print(data['AAPL'].head())
    """
    data_dict = {}
    
    for ticker in tickers:
        try:
            data_dict[ticker] = load_stock_data(ticker, start, end)
        except Exception as e:
            print(f"Failed to load {ticker}: {e}")
            
    return data_dict


def download_sample_data():
    """
    Download sample datasets for examples and tutorials.
    
    Downloads data for several stocks and saves to data/sample/ directory.
    This includes major tech stocks and market indices.
    """
    # Create sample data directory
    sample_dir = Path(__file__).parent.parent.parent / 'data' / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Define sample tickers
    tickers = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.',
        '^GSPC': 'S&P 500 Index',
        '^DJI': 'Dow Jones Industrial Average',
    }
    
    # Download 5 years of data
    start = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    
    print("Downloading sample datasets...")
    print("-" * 50)
    
    for ticker, name in tickers.items():
        try:
            print(f"\nDownloading {name} ({ticker})...")
            data = load_stock_data(ticker, start, end)
            
            # Save to CSV
            filename = sample_dir / f"{ticker.replace('^', '')}.csv"
            data.to_csv(filename)
            print(f"Saved to {filename}")
            
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
    
    print("\n" + "=" * 50)
    print("Sample data download complete!")
    print(f"Data saved to: {sample_dir}")


def load_sample_data(ticker: str) -> pd.DataFrame:
    """
    Load previously downloaded sample data.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'AAPL')
        
    Returns
    -------
    pd.DataFrame
        Stock data DataFrame
    """
    sample_dir = Path(__file__).parent.parent.parent / 'data' / 'sample'
    filename = sample_dir / f"{ticker.replace('^', '')}.csv"
    
    if not filename.exists():
        raise FileNotFoundError(
            f"Sample data for {ticker} not found. "
            f"Run download_sample_data() first."
        )
    
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    return data


def get_sp500_tickers() -> list:
    """
    Get list of S&P 500 ticker symbols.
    
    Returns
    -------
    list
        List of ticker symbols in the S&P 500
        
    Note
    ----
    This scrapes Wikipedia and may break if the page structure changes.
    """
    try:
        # Read S&P 500 constituents from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean up tickers (replace . with -)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        print(f"Found {len(tickers)} S&P 500 tickers")
        return tickers
    
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []


def add_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to stock data.
    
    This is a simple version. For more indicators, see preprocessing.py
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with OHLC columns
        
    Returns
    -------
    pd.DataFrame
        Data with additional technical indicator columns
    """
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    
    # Returns
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility (20-day rolling)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Stock Market Data Loader")
    print("=" * 50)
    
    # Download sample data
    download_sample_data()
    
    # Load and display Apple data
    print("\n" + "=" * 50)
    print("Loading AAPL data...")
    aapl = load_sample_data('AAPL')
    print(f"\nData shape: {aapl.shape}")
    print(f"\nFirst few rows:")
    print(aapl.head())
    
    # Add technical features
    aapl = add_technical_features(aapl)
    print(f"\nWith technical features:")
    print(aapl.head())
