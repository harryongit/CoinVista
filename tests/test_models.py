import pytest
import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ml_models import CryptoMLModels

class TestCryptoMLModels:
    @pytest.fixture
    def sample_time_series_data(self):
        """Create a sample time series dataset for testing"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200)
        prices = np.cumsum(np.random.normal(0, 1, 200)) + 40000
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'rolling_mean_7d': pd.Series(prices).rolling(window=7).mean(),
            'daily_return': pd.Series(prices).pct_change(),
            'crypto': 'bitcoin'
        }).dropna()

    @pytest.fixture
    def ml_models(self, sample_time_series_data):
        """Create CryptoMLModels instance with sample data"""
        return CryptoMLModels(sample_time_series_data)

    def test_init(self, ml_models):
        """Test initialization of CryptoMLModels"""
        assert hasattr(ml_models, 'data')
        assert hasattr(ml_models, 'features')
        assert hasattr(ml_models, 'target')

    def test_prepare_data(self, ml_models):
        """Test data preparation method"""
        X_train, X_test, y_train, y_test = ml_models.prepare_data()

        # Assertions
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[1] > 0

    def test_lstm_model(self, ml_models):
        """Test LSTM model creation and training"""
        X_train, X_test, y_train, y_test = ml_models.prepare_data()
        
        # Create and train LSTM model
        model = ml_models.create_lstm_model(X_train.shape[1])
        history = ml_models.train_lstm_model(model, X_train, y_train)

        # Assertions
        assert history is not None
        assert 'loss' in history.history
        assert 'val_loss' in history.history

    def test_model_prediction(self, ml_models):
        """Test model prediction capabilities"""
        X_train, X_test, y_train, y_test = ml_models.prepare_data()
        model = ml_models.create_lstm_model(X_train.shape[1])
        ml_models.train_lstm_model(model, X_train, y_train)

        # Make predictions
        predictions = ml_models.predict(model, X_test)

        # Assertions
        assert predictions.shape == y_test.shape
        assert not np.isnan(predictions).any()

    def test_model_evaluation(self, ml_models):
        """Test model performance evaluation"""
        X_train, X_test, y_train, y_test = ml_models.prepare_data()
        model = ml_models.create_lstm_model(X_train.shape[1])
        ml_models.train_lstm_model(model, X_train, y_train)

        # Make predictions
        predictions = ml_models.predict(model, X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Assertions
        assert mse >= 0
        assert mae >= 0
        assert rmse >= 0
        assert mse < 1e6  # Reasonable maximum error

    def test_time_series_cross_validation(self, ml_models):
        """Test time series cross-validation method"""
        cv_results = ml_models.time_series_cross_validation()

        # Assertions
        assert isinstance(cv_results, dict)
        assert 'mse_scores' in cv_results
        assert 'mae_scores' in cv_results
        assert len(cv_results['mse_scores']) > 0
        assert len(cv_results['mae_scores']) > 0

    def test_feature_importance(self, ml_models):
        """Test feature importance calculation"""
        X_train, X_test, y_train, y_test = ml_models.prepare_data()
        importances = ml_models.calculate_feature_importance(X_train, y_train)

        # Assertions
        assert isinstance(importances, np.ndarray)
        assert len(importances) == X_train.shape[1]
        assert np.all(importances >= 0)
        assert np.all(importances <= 1)
