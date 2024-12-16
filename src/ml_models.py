import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, 
    Dense, 
    Dropout, 
    GRU, 
    Conv1D, 
    MaxPooling1D, 
    Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class CryptoMLModels:
    def __init__(self, data_path=None):
        """
        Initialize Cryptocurrency Machine Learning Models
        
        Args:
            data_path (str): Path to processed cryptocurrency data
        """
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Load and preprocess data
        self.data = self._load_data(data_path)
        self.models = {}
        self.results = {}

    def _load_data(self, data_path=None):
        """
        Load processed cryptocurrency data
        
        Args:
            data_path (str): Optional path to data file
        
        Returns:
            pd.DataFrame: Processed cryptocurrency data
        """
        if data_path:
            return pd.read_csv(data_path, parse_dates=['timestamp'])
        
        # Default data loading logic
        processed_files = [f for f in os.listdir('data/processed') if f.endswith('.csv')]
        if not processed_files:
            raise FileNotFoundError("No processed data files found")
        
        latest_file = max(processed_files, key=lambda f: os.path.getctime(os.path.join('data/processed', f)))
        return pd.read_csv(os.path.join('data/processed', latest_file), parse_dates=['timestamp'])

    def prepare_data(self, crypto, lookback=30, forecast_horizon=7):
        """
        Prepare time series data for machine learning
        
        Args:
            crypto (str): Cryptocurrency identifier
            lookback (int): Number of historical days to use
            forecast_horizon (int): Number of days to predict
        
        Returns:
            Prepared feature and target arrays
        """
        # Filter data for specific cryptocurrency
        crypto_data = self.data[self.data['crypto'] == crypto].sort_values('timestamp')
        
        # Create features (lagged prices, rolling statistics)
        features = []
        targets = []
        
        # Create sliding window
        for i in range(len(crypto_data) - lookback - forecast_horizon + 1):
            # Extract lookback window
            window = crypto_data['price'].iloc[i:i+lookback]
            
            # Calculate additional features
            features.append([
                window.mean(),
                window.std(),
                window.min(),
                window.max(),
                window.quantile(0.25),
                window.quantile(0.75),
                *window.values  # Include individual lagged prices
            ])
            
            # Target is the average price of next forecast_horizon days
            target = crypto_data['price'].iloc[i+lookback:i+lookback+forecast_horizon].mean()
            targets.append(target)
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)
        
        return X, y

    def train_traditional_models(self, crypto):
        """
        Train traditional machine learning models
        
        Args:
            crypto (str): Cryptocurrency identifier
        
        Returns:
            Dictionary of trained models and their performances
        """
        # Prepare data
        X, y = self.prepare_data(crypto)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf')
        }
        
        # Train and evaluate models
        model_results = {}
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            model_results[name] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            }
        
        return model_results

    def train_deep_learning_models(self, crypto):
        """
        Train advanced deep learning models
        
        Args:
            crypto (str): Cryptocurrency identifier
        
        Returns:
            Dictionary of trained models and their performances
        """
        # Prepare data
        X, y = self.prepare_data(crypto)
        
        # Reshape for deep learning models
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Define deep learning models
        def create_lstm_model(input_shape):
            model = Sequential([
                LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            return model
        
        def create_cnn_lstm_model(input_shape):
            model = Sequential([
                Conv1D(64, 3, activation='relu', input_shape=input_shape),
                MaxPooling1D(2),
                LSTM(50, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            return model
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=0.00001
        )
        
        # Train LSTM
        lstm_model = create_lstm_model((X_train.shape[1], 1))
        lstm_history = lstm_model.fit(
            X_train, y_train, 
            epochs=100, 
            batch_size=32, 
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Train CNN-LSTM
        cnn_lstm_model = create_cnn_lstm_model((X_train.shape[1], 1))
        cnn_lstm_history = cnn_lstm_model.fit(
            X_train, y_train, 
            epochs=100, 
            batch_size=32, 
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate models
        def evaluate_model(model, X_test, y_test):
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            return {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            }
        
        # Store results
        deep_learning_results = {
            'LSTM': evaluate_model(lstm_model, X_test, y_test),
            'CNN-LSTM': evaluate_model(cnn_lstm_model, X_test, y_test)
        }
        
        return deep_learning_results

    def ensemble_prediction(self, crypto):
        """
        Create an ensemble prediction using multiple models
        
        Args:
            crypto (str): Cryptocurrency identifier
        
        Returns:
            Ensemble prediction results
        """
        # Prepare data
        X, y = self.prepare_data(crypto)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train multiple models
        models = [
            ('Linear', LinearRegression()),
            ('Ridge', Ridge()),
            ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]
        
        # Train models and store predictions
        model_predictions = []
        for name, model in models:
            model.fit(X_train, y_train)
            model_predictions.append(model.predict(X_test))
        
        # Ensemble prediction (simple average)
        ensemble_pred = np.mean(model_predictions, axis=0)
        
        # Evaluate ensemble
        ensemble_metrics = {
            'MSE': mean_squared_error(y_test, ensemble_pred),
            'MAE': mean_absolute_error(y_test, ensemble_pred),
            'R2': r2_score(y_test, ensemble_pred),
            'MAPE': mean_absolute_percentage_error(y_test, ensemble_pred)
        }
        
        return ensemble_metrics

    def run_comprehensive_analysis(self, cryptocurrencies=None):
        """
        Run comprehensive machine learning analysis for cryptocurrencies
        
        Args:
            cryptocurrencies (list): List of cryptocurrencies to analyze
        """
        # If no cryptocurrencies specified, use unique cryptos in dataset
        if cryptocurrencies is None:
            cryptocurrencies = self.data['crypto'].unique()
        
        # Comprehensive results storage
        comprehensive_results = {}
        
        for crypto in cryptocurrencies:
            print(f"Analyzing {crypto}...")
            
            # Traditional Models
            traditional_results = self.train_traditional_models(crypto)
            
            # Deep Learning Models
            deep_learning_results = self.train_deep_learning_models(crypto)
            
            # Ensemble Prediction
            ensemble_results = self.ensemble_prediction(crypto)
            
            # Store results
            comprehensive_results[crypto] = {
                'traditional_models': traditional_results,
                'deep_learning_models': deep_learning_results,
                'ensemble_prediction': ensemble_results
            }
        
        return comprehensive_results

def main():
    # Initialize ML Models
    ml_models = CryptoMLModels()
    
    # Run comprehensive analysis
    results = ml_models.run_comprehensive_analysis()
    
    # Print summary of results
    for crypto, crypto_results in results.items():
        print(f"\n{crypto.upper()} Analysis Results:")
        print("Traditional Models Performance:")
        for model, metrics in crypto_results['traditional_models'].items():
            print(f"{model}: {metrics}")
        
        print("\nDeep Learning Models Performance:")
        for model, metrics in crypto_results['deep_learning_models'].items():
            print(f"{model}: {metrics}")
        
        print("\nEnsemble Prediction Performance:")
        print(crypto_results['ensemble_prediction'])

if __name__ == "__main__":
    main()
