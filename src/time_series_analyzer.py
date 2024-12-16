import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import os

class CryptoTimeSeriesAnalyzer:
    def __init__(self, data_path: str = None):
        """
        Initialize Cryptocurrency Time Series Analyzer
        
        Args:
            data_path (str): Path to processed cryptocurrency data
        """
        self.processed_data_dir = 'data/processed'
        self.analysis_output_dir = os.path.join(self.processed_data_dir, 'analysis')
        
        # Create analysis output directory
        os.makedirs(self.analysis_output_dir, exist_ok=True)
        
        # Load data
        self.data = self._load_processed_data(data_path)

    def _load_processed_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load the processed cryptocurrency data
        
        Args:
            data_path (str): Optional specific data path
        
        Returns:
            pd.DataFrame: Loaded processed cryptocurrency data
        """
        if data_path:
            return pd.read_csv(data_path, parse_dates=['timestamp'])
        
        # Find the most recent processed CSV
        csv_files = [f for f in os.listdir(self.processed_data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No processed CSV files found")
        
        latest_file = max(csv_files, key=lambda f: os.path.getctime(os.path.join(self.processed_data_dir, f)))
        return pd.read_csv(os.path.join(self.processed_data_dir, latest_file), parse_dates=['timestamp'])

    def seasonal_decomposition(self, crypto: str):
        """
        Perform seasonal decomposition for a specific cryptocurrency
        
        Args:
            crypto (str): Cryptocurrency identifier
        """
        # Filter data for specific cryptocurrency
        crypto_data = self.data[self.data['crypto'] == crypto].set_index('timestamp')
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(crypto_data['price'], period=30)  # 30-day seasonality
        
        # Plotting
        plt.figure(figsize=(15, 10))
        
        plt.subplot(411)
        plt.title(f'{crypto.capitalize()} Time Series Decomposition')
        plt.plot(crypto_data.index, decomposition.observed)
        plt.ylabel('Observed')
        
        plt.subplot(412)
        plt.plot(crypto_data.index, decomposition.trend)
        plt.ylabel('Trend')
        
        plt.subplot(413)
        plt.plot(crypto_data.index, decomposition.seasonal)
        plt.ylabel('Seasonal')
        
        plt.subplot(414)
        plt.plot(crypto_data.index, decomposition.resid)
        plt.ylabel('Residual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_output_dir, f'{crypto}_decomposition.png'))
        plt.close()

    def prophet_forecast(self, crypto: str, periods: int = 30):
        """
        Create time series forecast using Prophet
        
        Args:
            crypto (str): Cryptocurrency identifier
            periods (int): Number of future periods to forecast
        """
        # Prepare data for Prophet
        crypto_data = self.data[self.data['crypto'] == crypto].copy()
        prophet_df = crypto_data[['timestamp', 'price']].rename(columns={'timestamp': 'ds', 'price': 'y'})
        
        # Create and fit Prophet model
        model = Prophet(
            daily_seasonality=True, 
            weekly_seasonality=True, 
            yearly_seasonality=True
        )
        model.fit(prophet_df)
        
        # Generate future predictions
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Plot forecast
        plt.figure(figsize=(15, 7))
        model.plot(forecast)
        plt.title(f'{crypto.capitalize()} Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.savefig(os.path.join(self.analysis_output_dir, f'{crypto}_forecast.png'))
        plt.close()
        
        # Save forecast to CSV
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_csv(
            os.path.join(self.analysis_output_dir, f'{crypto}_forecast.csv'), 
            index=False
        )

    def correlation_analysis(self):
        """
        Perform correlation analysis between cryptocurrencies
        """
        # Pivot data for correlation
        pivot_data = self.data.pivot(index='timestamp', columns='crypto', values='price')
        
        # Calculate correlation matrix
        correlation_matrix = pivot_data.corr()
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        plt.title('Cryptocurrency Price Correlation Matrix')
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_output_dir, 'correlation_matrix.png'))
        plt.close()
        
        # Save correlation matrix to CSV
        correlation_matrix.to_csv(os.path.join(self.analysis_output_dir, 'correlation_matrix.csv'))

def main():
    analyzer = CryptoTimeSeriesAnalyzer()
    
    # Analyze each cryptocurrency
    for crypto in analyzer.data['crypto'].unique():
        print(f"Analyzing {crypto}...")
        
        # Seasonal Decomposition
        analyzer.seasonal_decomposition(crypto)
        
        # Prophet Forecast
        analyzer.prophet_forecast(crypto)
    
    # Correlation Analysis
    analyzer.correlation_analysis()

if __name__ == "__main__":
    main()
```

<antArtifact identifier="machine-learning-models" type="application/vnd.ant.code" language="python" title="Cryptocurrency Price Prediction Models">
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

class CryptoPricePredictionModels:
    def __init__(self, data_path: str = None):
        """
        Initialize Cryptocurrency Price Prediction Models
        
        Args:
            data_path (str): Path to processed cryptocurrency data
        """
        self.processed_data_dir = 'data/processed'
        self.models_output_dir = os.path.join(self.processed_data_dir, 'models')
        
        # Create models output directory
        os.makedirs(self.models_output_dir, exist_ok=True)
        
        # Load data
        self.data = self._load_processed_data(data_path)
        
        # Random seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

    def _load_processed_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load the processed cryptocurrency data
        
        Args:
            data_path (str): Optional specific data path
        
        Returns:
            pd.DataFrame: Loaded processed cryptocurrency data
        """
        if data_path:
            return pd.read_csv(data_path, parse_dates=['timestamp'])
        
        # Find the most recent processed CSV
        csv_files = [f for f in os.listdir(self.processed_data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No processed CSV files found")
        
        latest_file = max(csv_files, key=lambda f: os.path.getctime(os.path.join(self.processed_data_dir, f)))
        return pd.read_csv(os.path.join(self.processed_data_dir, latest_file), parse_dates=['timestamp'])

    def prepare_lstm_data(self, crypto: str, lookback: int = 30) -> tuple:
        """
        Prepare data for LSTM model
        
        Args:
            crypto (str): Cryptocurrency identifier
            lookback (int): Number of previous days to use for prediction
        
        Returns:
            Tuple of prepared training and testing data
        """
        # Filter data for specific cryptocurrency
        crypto_data = self.data[self.data['crypto'] == crypto].set_index('timestamp')
        
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(crypto_data[['price']])
        
        # Create sequence data
        X, y = [], []
        for i in range(len(scaled_prices) - lookback):
            X.append(scaled_prices[i:i+lookback])
            y.append(scaled_prices[i+lookback])
        
        X, y = np.array(X), np.array(y)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        return X_train, X_test, y_train, y_test, scaler

    def build_lstm_model(self, input_shape: tuple) -> tf.keras.Model:
        """
        Build LSTM neural network model
        
        Args:
            input_shape (tuple): Shape of input data
        
        Returns:
            Compiled Keras LSTM model
        """
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_and_evaluate_model(self, crypto: str):
        """
        Train LSTM model and evaluate performance
        
        Args:
            crypto (str): Cryptocurrency identifier
        """
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = self.prepare_lstm_data(crypto)
        
        # Build model
        model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Train model
        history = model.fit(
            X_train, y_train, 
            epochs=50, 
            batch_size=32, 
            validation_split=0.2, 
            verbose=0
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        # Visualization
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_inv, label='Actual Prices', color='blue')
        plt.plot(y_pred_inv, label='Predicted Prices', color='red')
        plt.title(f'{crypto.capitalize()} Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.savefig(os.path.join(self.models_output_dir, f'{crypto}_prediction.png'))
        plt.close()
        
        # Save model performance metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'R2 Score'],
            'Value': [mse, mae, r2]
        })
        metrics_df.to_csv(os.path.join(self.models_output_dir, f'{crypto}_metrics.csv'), index=False)
        
        print(f"{crypto.capitalize()} Model Performance:")
        print(metrics_df)
        
        # Save model
        model.save(os.path.join(self.models_output_dir, f'{crypto}_model.h5'))

def main():
    predictor = CryptoPricePredictionModels()
    
    # Train and evaluate models for each cryptocurrency
    for crypto in predictor.data['crypto'].unique():
        print(f"Training model for {crypto}...")
        predictor.train_and_evaluate_model(crypto)

if __name__ == "__main__":
    main()
```

<antArtifact identifier="setup-py" type="application/vnd.ant.code" language="python" title="Project Setup Configuration">
from setuptools import setup, find_packages

setup(
    name='cryptocurrency-time-series-analysis',
    version='0.1.0',
    description='Comprehensive Time Series Analysis of Cryptocurrency Market Data',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=0.24.0',
        'requests>=2.26.0',
        'python-dotenv>=0.19.0',
        'statsmodels>=0.12.0',
        'prophet>=1.0.0',
        'tensorflow>=2.6.0',
        'keras>=2.6.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8',
)
