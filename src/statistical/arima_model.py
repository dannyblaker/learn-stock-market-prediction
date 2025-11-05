"""
ARIMA (AutoRegressive Integrated Moving Average) Model for Stock Prediction.

ARIMA is a classic statistical time series forecasting method that models
a series based on its own past values (autoregression), past forecast errors
(moving average), and differencing to make the series stationary (integration).

Best for: Short-term forecasting, understanding trends and seasonality
Pros: Interpretable, well-established, works well for stationary data
Cons: Assumes linear relationships, struggles with complex patterns
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class ARIMAPredictor:
    """
    ARIMA model for stock price prediction.
    
    Parameters
    ----------
    order : tuple
        (p, d, q) order of the ARIMA model
        - p: number of autoregressive terms
        - d: number of differences for stationarity
        - q: number of moving average terms
    
    Examples
    --------
    >>> model = ARIMAPredictor(order=(5, 1, 2))
    >>> model.fit(data['Close'])
    >>> predictions = model.predict(steps=30)
    """
    
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize ARIMA model.
        
        Parameters
        ----------
        order : tuple
            (p, d, q) parameters for ARIMA
            Default (1, 1, 1) is a good starting point
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.data = None
        
    def check_stationarity(self, timeseries, verbose=True):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Stationarity means statistical properties (mean, variance) don't change
        over time. Many time series models require stationary data.
        
        Parameters
        ----------
        timeseries : pd.Series
            Time series to test
        verbose : bool
            Print test results
            
        Returns
        -------
        bool
            True if stationary (p-value < 0.05), False otherwise
        """
        # Perform Augmented Dickey-Fuller test
        result = adfuller(timeseries.dropna())
        
        if verbose:
            print('ADF Statistic:', result[0])
            print('p-value:', result[1])
            print('Critical Values:')
            for key, value in result[4].items():
                print(f'\t{key}: {value}')
        
        # If p-value < 0.05, reject null hypothesis (non-stationary)
        # Therefore, series is stationary
        is_stationary = result[1] < 0.05
        
        if verbose:
            if is_stationary:
                print('\nSeries is stationary')
            else:
                print('\nSeries is NOT stationary - consider differencing')
        
        return is_stationary
    
    def find_optimal_order(self, timeseries, max_p=5, max_d=2, max_q=5):
        """
        Find optimal ARIMA order using AIC (Akaike Information Criterion).
        
        AIC balances model fit against complexity. Lower AIC = better model.
        
        Parameters
        ----------
        timeseries : pd.Series
            Time series data
        max_p : int
            Maximum p to try
        max_d : int
            Maximum d to try
        max_q : int
            Maximum q to try
            
        Returns
        -------
        tuple
            Optimal (p, d, q) order
        """
        print("Finding optimal ARIMA order...")
        print("This may take a few minutes...\n")
        
        best_aic = np.inf
        best_order = None
        
        # Grid search over parameters
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(timeseries, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            
                    except:
                        continue
        
        print(f"Optimal order: {best_order}")
        print(f"AIC: {best_aic:.2f}\n")
        
        return best_order
    
    def plot_diagnostics(self, timeseries):
        """
        Plot ACF and PACF to help determine ARIMA order.
        
        ACF (Autocorrelation Function): Shows correlation with lagged values
        PACF (Partial Autocorrelation Function): Shows direct effect of lags
        
        Rules of thumb:
        - AR(p): PACF cuts off after lag p, ACF decays
        - MA(q): ACF cuts off after lag q, PACF decays
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF plot
        plot_acf(timeseries.dropna(), lags=40, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        # PACF plot
        plot_pacf(timeseries.dropna(), lags=40, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        plt.show()
    
    def fit(self, data, find_order=False):
        """
        Fit ARIMA model to data.
        
        Parameters
        ----------
        data : pd.Series or np.ndarray
            Time series data (e.g., stock prices)
        find_order : bool
            If True, automatically find optimal order
        """
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        
        self.data = pd.Series(data) if isinstance(data, np.ndarray) else data
        
        # Find optimal order if requested
        if find_order:
            self.order = self.find_optimal_order(self.data)
        
        print(f"Fitting ARIMA{self.order} model...")
        
        # Create and fit model
        self.model = ARIMA(self.data, order=self.order)
        self.fitted_model = self.model.fit()
        
        print("Model fitted successfully!")
        print(f"\nModel Summary:")
        print(self.fitted_model.summary())
        
        return self
    
    def predict(self, steps=1):
        """
        Make future predictions.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
            
        Returns
        -------
        np.ndarray
            Forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Forecast
        forecast = self.fitted_model.forecast(steps=steps)
        
        return np.array(forecast)
    
    def predict_in_sample(self):
        """
        Get in-sample predictions (fitted values).
        
        Returns
        -------
        np.ndarray
            Predictions for training data
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.fitted_model.fittedvalues
    
    def plot_forecast(self, steps=30, data=None):
        """
        Plot historical data and forecast.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        data : pd.Series, optional
            Historical data to plot (uses training data if None)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        if data is None:
            data = self.data
        
        # Make forecast
        forecast = self.predict(steps=steps)
        
        # Create forecast index
        if isinstance(data.index, pd.DatetimeIndex):
            freq = pd.infer_freq(data.index)
            forecast_index = pd.date_range(
                start=data.index[-1], 
                periods=steps + 1, 
                freq=freq
            )[1:]
        else:
            forecast_index = range(len(data), len(data) + steps)
        
        # Plot
        plt.figure(figsize=(14, 7))
        
        # Historical data
        plt.plot(data.index, data, label='Historical', linewidth=2)
        
        # Fitted values
        fitted = self.fitted_model.fittedvalues
        plt.plot(fitted.index, fitted, label='Fitted', 
                linewidth=1.5, alpha=0.7, linestyle='--')
        
        # Forecast
        plt.plot(forecast_index, forecast, 
                label=f'{steps}-step Forecast', 
                linewidth=2, color='red')
        
        # Confidence interval (approximate)
        std_error = np.std(data - fitted)
        ci_lower = forecast - 1.96 * std_error
        ci_upper = forecast + 1.96 * std_error
        plt.fill_between(forecast_index, ci_lower, ci_upper, 
                        alpha=0.2, color='red')
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'ARIMA{self.order} Forecast')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def example_usage():
    """Example of using ARIMA for stock prediction."""
    from ..utils.data_loader import load_stock_data
    from ..utils.evaluation import evaluate_regression
    
    print("=" * 60)
    print("ARIMA Example: Stock Price Forecasting")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data = load_stock_data('AAPL', start='2020-01-01', end='2023-12-31')
    prices = data['Close']
    
    # Split data (80% train, 20% test)
    train_size = int(len(prices) * 0.8)
    train_data = prices[:train_size]
    test_data = prices[train_size:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create and train model
    print("\n2. Checking stationarity...")
    model = ARIMAPredictor(order=(5, 1, 2))
    model.check_stationarity(train_data)
    
    print("\n3. Training ARIMA model...")
    model.fit(train_data)
    
    # Make predictions
    print("\n4. Making predictions...")
    predictions = []
    
    # Rolling forecast (predict one step at a time)
    for i in range(len(test_data)):
        pred = model.predict(steps=1)
        predictions.append(pred[0])
        
        # Update model with actual value (rolling forecast)
        if i < len(test_data) - 1:
            new_data = pd.concat([train_data, test_data[:i+1]])
            model.fit(new_data)
    
    predictions = np.array(predictions)
    
    # Evaluate
    print("\n5. Evaluation:")
    metrics = evaluate_regression(test_data.values, predictions)
    
    # Plot results
    print("\n6. Plotting results...")
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index, test_data.values, 
            label='Actual', linewidth=2)
    plt.plot(test_data.index, predictions, 
            label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title('ARIMA Stock Price Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Future forecast
    print("\n7. Future forecast...")
    model.fit(prices)  # Refit on all data
    model.plot_forecast(steps=30)
    
    print("\n" + "=" * 60)
    print("ARIMA Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
