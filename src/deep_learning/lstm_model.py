"""
LSTM (Long Short-Term Memory) Model for Stock Prediction.

LSTM is a type of recurrent neural network (RNN) that can learn long-term
dependencies in sequential data. It's particularly well-suited for time series
like stock prices where patterns may span many time steps.

Best for: Learning complex sequential patterns, multi-step forecasting
Pros: Captures long-term dependencies, handles non-linear patterns
Cons: Requires lots of data, computationally expensive, hard to interpret
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class LSTMModel(nn.Module):
    """
    LSTM neural network for time series prediction.
    
    Architecture:
    - Input layer
    - LSTM layers (1 or more)
    - Dropout for regularization
    - Fully connected output layer
    """
    
    def __init__(self, input_size, hidden_size=50, num_layers=2, 
                 output_size=1, dropout=0.2):
        """
        Initialize LSTM model.
        
        Parameters
        ----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of LSTM units in each layer
        num_layers : int
            Number of LSTM layers
        output_size : int
            Number of output values (1 for regression)
        dropout : float
            Dropout rate between LSTM layers
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns
        -------
        torch.Tensor
            Output predictions
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take output from last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out


class LSTMPredictor:
    """
    Wrapper class for training and using LSTM models.
    
    Parameters
    ----------
    input_size : int
        Number of features
    hidden_size : int
        Size of LSTM hidden state
    num_layers : int
        Number of LSTM layers
    sequence_length : int
        Length of input sequences (e.g., 60 days)
    learning_rate : float
        Learning rate for optimizer
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    device : str
        'cuda' or 'cpu'
    """
    
    def __init__(self, input_size, hidden_size=50, num_layers=2,
                 sequence_length=60, learning_rate=0.001, 
                 epochs=50, batch_size=32, device=None):
        """Initialize LSTM predictor."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def prepare_data(self, data, target_column=0):
        """
        Prepare sequential data for LSTM.
        
        Parameters
        ----------
        data : np.ndarray
            2D array of shape (samples, features)
        target_column : int
            Index of target column
            
        Returns
        -------
        X : np.ndarray
            Sequences of shape (samples, sequence_length, features)
        y : np.ndarray
            Targets
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, target_column])
        
        return np.array(X), np.array(y)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train LSTM model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training sequences (samples, sequence_length, features)
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray, optional
            Validation sequences
        y_val : np.ndarray, optional
            Validation targets
        verbose : bool
            Print training progress
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        print(f"\nTraining LSTM model for {self.epochs} epochs...")
        print("-" * 60)
        
        # Training loop
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = self.criterion(val_predictions, y_val_tensor)
                    self.history['val_loss'].append(val_loss.item())
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {val_loss.item():.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - "
                          f"Train Loss: {avg_train_loss:.6f}")
        
        print("-" * 60)
        print("Training complete!")
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequences
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy().flatten()
    
    def plot_training_history(self):
        """Plot training and validation loss."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def example_usage():
    """Example of using LSTM for stock prediction."""
    from ..utils.data_loader import load_stock_data
    from ..utils.preprocessing import (
        calculate_technical_indicators,
        train_test_split_temporal,
        scale_features
    )
    from ..utils.evaluation import evaluate_regression
    
    print("=" * 60)
    print("LSTM Example: Stock Price Prediction")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading and preparing data...")
    data = load_stock_data('AAPL', start='2018-01-01', end='2023-12-31')
    
    # Add technical indicators
    data = calculate_technical_indicators(data)
    
    # Select features
    feature_cols = ['Close', 'Volume', 'RSI', 'MACD', 'SMA_20', 'SMA_50']
    data = data[feature_cols].dropna()
    
    print(f"Data shape: {data.shape}")
    
    # Split data
    train_data, val_data, test_data = train_test_split_temporal(
        data, test_size=0.2, validation_size=0.1
    )
    
    # Scale features
    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_data, val_data, test_data, method='minmax'
    )
    
    # Create LSTM predictor
    sequence_length = 60  # Use 60 days to predict next day
    
    model = LSTMPredictor(
        input_size=len(feature_cols),
        hidden_size=50,
        num_layers=2,
        sequence_length=sequence_length,
        learning_rate=0.001,
        epochs=50,
        batch_size=32
    )
    
    # Prepare sequences
    print("\n2. Preparing sequences...")
    X_train, y_train = model.prepare_data(train_scaled.values)
    X_val, y_val = model.prepare_data(val_scaled.values)
    X_test, y_test = model.prepare_data(test_scaled.values)
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    # Train model
    print("\n3. Training LSTM model...")
    model.fit(X_train, y_train, X_val, y_val)
    
    # Plot training history
    model.plot_training_history()
    
    # Make predictions
    print("\n4. Making predictions...")
    predictions = model.predict(X_test)
    
    # Inverse transform predictions
    # (predictions are scaled, need to convert back to original scale)
    predictions_full = np.zeros((len(predictions), len(feature_cols)))
    predictions_full[:, 0] = predictions
    predictions_original = scaler.inverse_transform(predictions_full)[:, 0]
    
    y_test_full = np.zeros((len(y_test), len(feature_cols)))
    y_test_full[:, 0] = y_test
    y_test_original = scaler.inverse_transform(y_test_full)[:, 0]
    
    # Evaluate
    print("\n5. Evaluation:")
    metrics = evaluate_regression(y_test_original, predictions_original)
    
    # Plot results
    print("\n6. Plotting results...")
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_original, label='Actual', linewidth=2)
    plt.plot(predictions_original, label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Price ($)')
    plt.title('LSTM Stock Price Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("LSTM Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
