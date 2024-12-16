import pandas as pd
import numpy as np
from typing import List, Dict

class CryptoFeatureEngineer:
    def __init__(self, data_path: str = None):
        """
        Initialize Cryptocurrency Feature Engineering
        
        Args:
            data_path (str): Path to processed cryptocurrency data
        """
        self.processed_data_dir = 'data/processed'
        self.data = self._load_processed_data(data_path)

    def _load_processed_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load processed cryptocurrency data
        
        Args:
            data_path (str): Optional specific data path
        
        Returns:
            pd.DataFrame: Loaded processed cryptocurrency data
        """
        import os
        
        if data_path:
            return pd.read_csv(data_path, parse_dates=['timestamp'])
        
        # Find the most recent processed CSV
        csv_files = [f for f in os.listdir(self.processed_data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No processed CSV files found")
        
        latest_file = max(csv_files, key=lambda f: os.path.getctime(os.path.join(self.processed_data_dir, f)))
        return pd.read_csv(os.path.join(self.processed_data_dir, latest_file), parse_dates=['timestamp'])

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """
        Calculate advanced technical indicators for cryptocurrencies
        
        Returns:
            pd.DataFrame: DataFrame with additional technical indicators
        """
        def calculate_crypto_indicators(group):
            # Relative Strength Index (RSI)
            group['rsi_14'] = self._calculate_rsi(group['price'], window=14)
            
            # Moving Average Convergence Divergence (MACD)
            group['macd'], group['signal_line'] = self._calculate_macd(group['price'])
            
            # Bollinger Bands
            group['bollinger_middle'], group['bollinger_upper'], group['bollinger_lower'] = self._calculate_bollinger_bands(group['price'])
            
            # On-Balance Volume (OBV) - simulated with price changes
            group['obv'] = self._calculate_obv(group['price'])
            
            return group

        # Apply indicators to each cryptocurrency group
        featured_data = self.data.groupby('crypto').apply(calculate_crypto_indicators).reset_index(drop=True)
        
        return featured_data

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices (pd.Series): Price series
            window (int): RSI calculation window
        
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi

    def _calculate_macd(self, prices: pd.Series, 
                         fast_period: int = 12, 
                         slow_period: int = 26, 
                         signal_period: int = 9) -> tuple:
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            prices (pd.Series): Price series
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
        
        Returns:
            tuple: MACD line and signal line
        """
        # Exponential Moving Averages
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        
        # MACD Line
        macd_line = fast_ema - slow_ema
        
        # Signal Line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        return macd_line, signal_line

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """
        Calculate Bollinger Bands
        
        Args:
            prices (pd.Series): Price series
            window (int): Moving average window
            num_std (float): Number of standard deviations
        
        Returns:
            tuple: Middle, upper, and lower Bollinger Bands
        """
        middle_band = prices.rolling(window=window).mean()
        std_dev = prices.rolling(window=window).std()
        
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return middle_band, upper_band, lower_band

    def _calculate_obv(self, prices: pd.Series) -> pd.Series:
        """
        Simulate On-Balance Volume (OBV)
        
        Args:
            prices (pd.Series): Price series
        
        Returns:
            pd.Series: Simulated OBV values
        """
        # Simulate OBV with price changes
        price_changes = prices.diff()
        obv = (price_changes >= 0).astype(int).cumsum() - (price_changes < 0).astype(int).cumsum()
        
        return obv

    def create_lagged_features(self, lags: List[int] = [1, 3, 7, 14, 30]) -> pd.DataFrame:
        """
        Create lagged price features
        
        Args:
            lags (List[int]): List of lag periods
        
        Returns:
            pd.DataFrame: DataFrame with lagged features
        """
        def create_lag_features(group):
            for lag in lags:
                group[f'price_lag_{lag}'] = group['price'].shift(lag)
                group[f'return_lag_{lag}'] = group['price'].pct_change(lag)
            return group

        lagged_data = self.data.groupby('crypto').apply(create_lag_features).reset_index(drop=True)
        return lagged_data

    def generate_feature_summary(self) -> Dict:
        """
        Generate summary of engineered features
        
        Returns:
            Dict: Summary of feature engineering results
        """
        # Technical Indicators
        technical_data = self.calculate_technical_indicators()
        
        # Lagged Features
        lagged_data = self.create_lagged_features()
        
        # Combine features
        combined_features = pd.merge(technical_data, lagged_data, on=['timestamp', 'crypto', 'price'])
        
        # Feature summary
        feature_summary = {
            'total_features': len(combined_features.columns),
            'cryptocurrencies': combined_features['crypto'].unique().tolist(),
            'date_range': {
                'start': combined_features['timestamp'].min(),
                'end': combined_features['timestamp'].max()
            },
            'feature_types': [
                'price', 
                'technical_indicators', 
                'lagged_features', 
                'returns'
            ]
        }
        
        # Save combined features
        combined_features.to_csv(f'{self.processed_data_dir}/combined_features.csv', index=False)
        
        return feature_summary

def main():
    feature_engineer = CryptoFeatureEngineer()
    
    # Generate features
    technical_indicators = feature_engineer.calculate_technical_indicators()
    lagged_features = feature_engineer.create_lagged_features()
    
    # Generate feature summary
    feature_summary = feature_engineer.generate_feature_summary()
    
    # Print feature summary
    print("Feature Engineering Summary:")
    for key, value in feature_summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
