"""
Random Forest model for stock market prediction.

Random Forest is an ensemble learning method that constructs multiple decision
trees and combines their predictions. It's robust, handles non-linear relationships
well, and provides feature importance insights.

Best for: Classification (up/down), feature importance analysis, baseline models
Pros: Handles non-linearity, robust to overfitting, feature importance
Cons: Can be slow, less interpretable than single trees
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class RandomForestPredictor:
    """
    Random Forest model for stock prediction.
    
    Can perform both regression (predicting price) and classification
    (predicting direction: up/down).
    
    Parameters
    ----------
    task : str
        'regression' or 'classification'
    n_estimators : int
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of trees
    random_state : int
        Random seed for reproducibility
        
    Examples
    --------
    >>> model = RandomForestPredictor(task='classification', n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, task='regression', n_estimators=100, 
                 max_depth=None, random_state=42):
        """Initialize Random Forest model."""
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        if task == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1  # Use all CPU cores
            )
        elif task == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError("task must be 'regression' or 'classification'")
        
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X, y, feature_names=None):
        """
        Fit Random Forest model.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix (samples x features)
        y : np.ndarray or pd.Series
            Target values
        feature_names : list, optional
            Names of features (for interpretation)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = feature_names
        
        print(f"Training Random Forest {self.task} model...")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        print("Model trained successfully!")
        
        # Print training score
        train_score = self.model.score(X, y)
        print(f"Training {self.task} score: {train_score:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities (classification only).
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance scores.
        
        Feature importance shows which features contribute most to predictions.
        Based on how much each feature decreases impurity in the trees.
        
        Parameters
        ----------
        top_n : int
            Number of top features to return
            
        Returns
        -------
        pd.DataFrame
            Feature importance rankings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importances = self.model.feature_importances_
        
        if self.feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance.
        
        Parameters
        ----------
        top_n : int
            Number of top features to plot
        """
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['Importance'])
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance - Random Forest')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    
    def tune_hyperparameters(self, X, y, cv=3):
        """
        Tune hyperparameters using GridSearchCV.
        
        This searches over different parameter combinations to find the best.
        WARNING: Can be time-consuming!
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training features
        y : np.ndarray or pd.Series
            Training targets
        cv : int
            Number of cross-validation folds
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        print("Tuning hyperparameters (this may take a while)...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        if self.task == 'regression':
            base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        else:
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, 
            scoring='r2' if self.task == 'regression' else 'accuracy',
            verbose=1, n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        print("\nBest parameters:")
        print(grid_search.best_params_)
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return grid_search.best_params_


def example_usage():
    """Example of using Random Forest for stock prediction."""
    from ..utils.data_loader import load_stock_data
    from ..utils.preprocessing import (
        calculate_technical_indicators,
        create_lagged_features,
        create_target_variable,
        train_test_split_temporal
    )
    from ..utils.evaluation import evaluate_regression, evaluate_classification
    
    print("=" * 60)
    print("Random Forest Example: Stock Price Prediction")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    data = load_stock_data('AAPL', start='2020-01-01', end='2023-12-31')
    
    # Add technical indicators
    data = calculate_technical_indicators(data)
    
    # Add lagged features
    data = create_lagged_features(
        data, 
        columns=['Close', 'Volume', 'RSI', 'MACD'],
        lags=[1, 2, 3, 5, 10]
    )
    
    # Create target (next day's direction)
    data = create_target_variable(data, target_type='direction', horizon=1)
    
    # Remove NaN values
    data = data.dropna()
    
    print(f"Total samples: {len(data)}")
    print(f"Features: {data.shape[1] - 1}")  # -1 for target
    
    # Prepare features and target
    feature_cols = [col for col in data.columns if col != 'Target']
    X = data[feature_cols]
    y = data['Target']
    
    # Split data
    train_data, val_data, test_data = train_test_split_temporal(
        data, test_size=0.2, validation_size=0.1
    )
    
    X_train = train_data[feature_cols]
    y_train = train_data['Target']
    X_test = test_data[feature_cols]
    y_test = test_data['Target']
    
    # Train model (classification)
    print("\n2. Training Random Forest Classifier...")
    model = RandomForestPredictor(
        task='classification',
        n_estimators=100,
        max_depth=20
    )
    model.fit(X_train, y_train, feature_names=feature_cols)
    
    # Make predictions
    print("\n3. Making predictions...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Evaluate
    print("\n4. Evaluation:")
    metrics = evaluate_classification(y_test.values, predictions)
    
    # Feature importance
    print("\n5. Top 15 Most Important Features:")
    importance_df = model.get_feature_importance(top_n=15)
    print(importance_df.to_string(index=False))
    
    print("\n6. Plotting feature importance...")
    model.plot_feature_importance(top_n=15)
    
    print("\n" + "=" * 60)
    print("Random Forest Example Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("- Random Forest handles non-linear patterns well")
    print("- Feature importance helps understand what drives predictions")
    print("- Ensemble method reduces overfitting risk")
    print("- Works well as a baseline before trying complex models")


if __name__ == "__main__":
    example_usage()
