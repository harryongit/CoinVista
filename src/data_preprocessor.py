import pandas as pd
import numpy as np
from typing import Tuple, Dict
import os
import matplotlib.pyplot as plt
import seaborn as sns

class CryptoDataPreprocessor:
    def __init__(self, data_path: str = None):
        """
        Initialize Cryptocurrency Data Preprocessor
        
        Args:
            data_path (str): Path to raw cryptocurrency data
        """
        self.raw_data_dir = 'data/raw'
        self.processed_data_dir = 'data/processed'
        
        # Create processed data directory
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Load data
        self.data = self._load_latest_data(data_path)

    def _load_latest_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load the most recent CSV file from raw data directory
        
        Args:
            data_path (str): Optional specific data path
        
        Returns:
            pd.DataFrame: Loaded cryptocurrency price data
        """
        if data_path:
            return pd.read_csv(data_path)
        
        # Find the most recent CSV in raw data directory
        csv_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in raw data directory")
        
        latest_file = max(csv_files, key=lambda f: os.path.getctime(os.path.join(self.raw_data_dir, f)))
        return pd.read_csv(os.path.join(self.raw_data_dir, latest_file))

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess cryptocurrency price data
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Convert timestamp to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Sort by timestamp
        self.data = self.data.sort_values('timestamp')
        
        # Set timestamp as index
        self.data.set_index('timestamp', inplace=True)
        
        # Group by cryptocurrency and resample to daily data
        processed_data = self.data.groupby('crypto')['price'].resample('D').mean().reset_index()
        
        return processed_data

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional time series features
        
        Args:
            data (pd.DataFrame): Preprocessed price data
        
        Returns:
            pd.DataFrame: Data with additional features
        """
        # Calculate rolling statistics
        features_data = data.copy()
        features_data['rolling_mean_7d'] = features_data.groupby('crypto')['price'].rolling(window=7).mean().reset_index(0, drop=True)
        features_data['rolling_std_7d'] = features_data.groupby('crypto')['price'].rolling(window=7).std().reset_index(0, drop=True)
        
        # Calculate percentage change
        features_data['daily_return'] = features_data.groupby('crypto')['price'].pct_change()
        
        # Calculate cumulative returns
        features_data['cumulative_return'] = features_data.groupby('crypto')['daily_return'].cumsum()
        
        return features_data

    def detect_outliers(self, data: pd.DataFrame) -> Dict:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Args:
            data (pd.DataFrame): Input data
        
        Returns:
            Dict: Outliers for each cryptocurrency
        """
        outliers = {}
        
        for crypto in data['crypto'].unique():
            crypto_data = data[data['crypto'] == crypto]
            
            Q1 = crypto_data['price'].quantile(0.25)
            Q3 = crypto_data['price'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            crypto_outliers = crypto_data[(crypto_data['price'] < lower_bound) | (crypto_data['price'] > upper_bound)]
            outliers[crypto] = crypto_outliers
        
        return outliers

    def visualize_preprocessing(self, processed_data: pd.DataFrame):
        """
        Create visualizations of preprocessed data
        
        Args:
            processed_data (pd.DataFrame): Preprocessed cryptocurrency data
        """
        plt.figure(figsize=(15, 10))
        
        # Price trends
        plt.subplot(2, 2, 1)
        for crypto in processed_data['crypto'].unique():
            crypto_data = processed_data[processed_data['crypto'] == crypto]
            plt.plot(crypto_data['timestamp'], crypto_data['price'], label=crypto)
        plt.title('Cryptocurrency Price Trends')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        
        # Daily Returns Distribution
        plt.subplot(2, 2, 2)
        processed_data['daily_return'] = processed_data.groupby('crypto')['price'].pct_change()
        sns.boxplot(x='crypto', y='daily_return', data=processed_data)
        plt.title('Daily Returns Distribution')
        plt.ylabel('Daily Return')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{self.processed_data_dir}/preprocessing_visualization.png')
        plt.close()

    def save_processed_data(self, data: pd.DataFrame):
        """
        Save processed data to CSV
        
        Args:
            data (pd.DataFrame): Processed cryptocurrency data
        """
        filename = f'{self.processed_data_dir}/processed_crypto_data.csv'
        data.to_csv(filename, index=False)
        print(f"Processed data saved to {filename}")

def main():
    preprocessor = CryptoDataPreprocessor()
    
    # Preprocess data
    preprocessed_data = preprocessor.preprocess_data()
    
    # Calculate features
    featured_data = preprocessor.calculate_features(preprocessed_data)
    
    # Detect outliers
    outliers = preprocessor.detect_outliers(featured_data)
    print("Outliers detected for each cryptocurrency:")
    for crypto, crypto_outliers in outliers.items():
        print(f"{crypto}: {len(crypto_outliers)} outliers")
    
    # Visualize preprocessing
    preprocessor.visualize_preprocessing(featured_data)
    
    # Save processed data
    preprocessor.save_processed_data(featured_data)

if __name__ == "__main__":
    main()
```

<antArtifact identifier="requirements" type="text/markdown" title="Project Requirements">
# Project Requirements

## Python Dependencies

### Data Manipulation and Analysis
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=0.24.0

### Data Collection
- requests>=2.26.0
- python-dotenv>=0.19.0

### Time Series Analysis
- statsmodels>=0.12.0
- prophet>=1.0.0

### Machine Learning
- tensorflow>=2.6.0
- keras>=2.6.0

### Visualization
- matplotlib>=3.4.0
- seaborn>=0.11.0

### Notebook Environment
- jupyter>=1.0.0
- notebook>=6.4.0

### Development and Testing
- pytest>=6.2.0
- flake
