"""
Trading Environment for Reinforcement Learning.

This module provides a gym-compatible environment for training RL agents
to learn trading strategies. The agent observes market state and takes
actions (buy, sell, hold) to maximize profit.

Best for: Learning complete trading strategies, optimizing entry/exit
Pros: Learns optimal actions considering risk, adapts to changing conditions
Cons: Very data-hungry, computationally expensive, hard to debug
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib.pyplot as plt


class TradingEnv(gym.Env):
    """
    Stock trading environment for reinforcement learning.
    
    The agent can take three actions: Buy (0), Sell (1), or Hold (2)
    
    Observation space: Market features (prices, indicators, etc.)
    Action space: Discrete(3) - Buy, Sell, Hold
    
    Reward: Profit/loss from trading decisions
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with OHLC and features
    initial_balance : float
        Starting cash
    transaction_cost : float
        Cost per transaction (as fraction)
    window_size : int
        Number of time steps in observation
    """
    
    def __init__(self, data, initial_balance=10000, 
                 transaction_cost=0.001, window_size=10):
        """Initialize trading environment."""
        super(TradingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        # Extract features (everything except OHLC)
        self.feature_columns = [col for col in data.columns 
                               if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        self.features = data[self.feature_columns].values
        self.prices = data['Close'].values
        
        # Action space: 0=Buy, 1=Sell, 2=Hold
        self.action_space = spaces.Discrete(3)
        
        # Observation space: window of features + position info
        n_features = len(self.feature_columns) * window_size + 3  # +3 for position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(n_features,), 
            dtype=np.float32
        )
        
        # Episode variables
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_value = initial_balance
        
        # Trading history
        self.trades = []
        self.portfolio_values = []
    
    def reset(self):
        """
        Reset environment to initial state.
        
        Returns
        -------
        observation : np.ndarray
            Initial observation
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_value = self.initial_balance
        
        self.trades = []
        self.portfolio_values = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get current observation.
        
        Returns
        -------
        observation : np.ndarray
            Current market state + position information
        """
        # Get window of features
        start = self.current_step - self.window_size
        end = self.current_step
        window_features = self.features[start:end].flatten()
        
        # Add position information
        current_price = self.prices[self.current_step]
        position_value = self.shares_held * current_price
        total_value = self.balance + position_value
        
        position_info = np.array([
            self.shares_held / 100,  # Normalized shares
            self.balance / self.initial_balance,  # Normalized balance
            position_value / self.initial_balance  # Normalized position value
        ])
        
        observation = np.concatenate([window_features, position_info])
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """
        Execute one time step.
        
        Parameters
        ----------
        action : int
            0=Buy, 1=Sell, 2=Hold
            
        Returns
        -------
        observation : np.ndarray
            Next observation
        reward : float
            Reward for this action
        done : bool
            Whether episode is finished
        info : dict
            Additional information
        """
        current_price = self.prices[self.current_step]
        previous_value = self.balance + (self.shares_held * current_price)
        
        # Execute action
        if action == 0:  # Buy
            # Buy as many shares as possible with current balance
            shares_to_buy = int(self.balance / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                transaction_fee = cost * self.transaction_cost
                
                self.balance -= (cost + transaction_fee)
                self.shares_held += shares_to_buy
                self.total_shares_bought += shares_to_buy
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'value': self.balance + (self.shares_held * current_price)
                })
        
        elif action == 1:  # Sell
            # Sell all held shares
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price
                transaction_fee = proceeds * self.transaction_cost
                
                self.balance += (proceeds - transaction_fee)
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': self.shares_held,
                    'price': current_price,
                    'value': self.balance
                })
                
                self.shares_held = 0
        
        # else: action == 2 (Hold) - do nothing
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        if self.current_step < len(self.prices):
            current_price = self.prices[self.current_step]
        
        current_value = self.balance + (self.shares_held * current_price)
        self.portfolio_values.append(current_value)
        
        # Calculate reward (change in portfolio value)
        reward = current_value - previous_value
        
        # Check if episode is done
        done = self.current_step >= len(self.prices) - 1
        
        # Get next observation
        if done:
            observation = self._get_observation()
        else:
            observation = self._get_observation()
        
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': current_value,
            'total_profit': current_value - self.initial_balance
        }
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """
        Render environment state.
        
        Parameters
        ----------
        mode : str
            Rendering mode
        """
        current_price = self.prices[self.current_step]
        current_value = self.balance + (self.shares_held * current_price)
        profit = current_value - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        
        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares: {self.shares_held}")
        print(f"Portfolio Value: ${current_value:.2f}")
        print(f"Profit: ${profit:.2f} ({profit_pct:.2f}%)")
        print("-" * 40)
    
    def plot_portfolio(self):
        """Plot portfolio value over time."""
        plt.figure(figsize=(14, 7))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_values, label='Portfolio Value', linewidth=2)
        plt.axhline(y=self.initial_balance, color='r', 
                   linestyle='--', label='Initial Balance')
        plt.ylabel('Portfolio Value ($)')
        plt.title('RL Trading Agent Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot price with buy/sell markers
        plt.subplot(2, 1, 2)
        start_idx = self.window_size
        plt.plot(range(start_idx, len(self.prices)), 
                self.prices[start_idx:], 
                label='Price', linewidth=1.5, alpha=0.7)
        
        # Mark trades
        for trade in self.trades:
            if trade['action'] == 'buy':
                plt.scatter(trade['step'], trade['price'], 
                          color='green', marker='^', s=100, 
                          label='Buy' if trade == self.trades[0] else '')
            elif trade['action'] == 'sell':
                plt.scatter(trade['step'], trade['price'], 
                          color='red', marker='v', s=100,
                          label='Sell' if 'Sell' not in plt.gca().get_legend_handles_labels()[1] else '')
        
        plt.ylabel('Price ($)')
        plt.xlabel('Time Step')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def example_usage():
    """Example of using the trading environment."""
    from ..utils.data_loader import load_stock_data
    from ..utils.preprocessing import calculate_technical_indicators
    
    print("=" * 60)
    print("RL Trading Environment Example")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading data...")
    data = load_stock_data('AAPL', start='2022-01-01', end='2023-12-31')
    data = calculate_technical_indicators(data)
    
    # Select features for observation
    feature_cols = ['Close', 'RSI', 'MACD', 'SMA_20', 'Volume_Ratio']
    data = data[feature_cols].dropna()
    
    print(f"Data shape: {data.shape}")
    
    # Create environment
    print("\n2. Creating trading environment...")
    env = TradingEnv(
        data=data,
        initial_balance=10000,
        transaction_cost=0.001,
        window_size=10
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test with random agent
    print("\n3. Testing with random agent...")
    obs = env.reset()
    done = False
    
    while not done:
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    
    # Print results
    print("\n4. Results:")
    final_value = info['portfolio_value']
    profit = info['total_profit']
    profit_pct = (profit / env.initial_balance) * 100
    
    print(f"Initial Balance: ${env.initial_balance:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Profit: ${profit:.2f} ({profit_pct:.2f}%)")
    print(f"Total Trades: {len(env.trades)}")
    
    # Plot portfolio
    print("\n5. Plotting portfolio performance...")
    env.plot_portfolio()
    
    print("\n" + "=" * 60)
    print("Note: This is a random agent for demonstration.")
    print("Train a DQN or PPO agent for intelligent trading!")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
