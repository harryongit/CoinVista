import requests
import pandas as pd
import time
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CryptoDataCollector:
    def __init__(self, 
                 cryptocurrencies: List[str] = ['bitcoin', 'ethereum', 'binancecoin'],
                 days: int = 365):
        """
        Initialize Cryptocurrency Data Collector
        
        Args:
            cryptocurrencies (List[str]): List of cryptocurrency ids
            days (int): Number of historical days to collect
        """
        self.base_url = "https://api.coingecko.com/api/v3"
        self.cryptocurrencies = cryptocurrencies
        self.days = days
        self.data_dir = 'data/raw'
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_historical_data(self, crypto_id: str) -> pd.DataFrame:
        """
        Fetch historical price data for a specific cryptocurrency
        
        Args:
            crypto_id (str): CoinGecko cryptocurrency id
        
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            url = f"{self.base_url}/coins/{crypto_id}/market_chart?vs_currency=usd&days={self.days}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract prices
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
            prices_df['crypto'] = crypto_id
            
            return prices_df
        
        except requests.RequestException as e:
            print(f"Error fetching data for {crypto_id}: {e}")
            return pd.DataFrame()

    def collect_multi_crypto_data(self) -> pd.DataFrame:
        """
        Collect historical data for multiple cryptocurrencies
        
        Returns:
            pd.DataFrame: Combined historical price data
        """
        all_crypto_data = []
        
        for crypto in self.cryptocurrencies:
            print(f"Collecting data for {crypto}...")
            crypto_data = self.fetch_historical_data(crypto)
            
            if not crypto_data.empty:
                all_crypto_data.append(crypto_data)
                time.sleep(1)  # Rate limiting
        
        combined_data = pd.concat(all_crypto_data, ignore_index=True)
        
        # Save to CSV
        filename = f"{self.data_dir}/crypto_prices_{int(time.time())}.csv"
        combined_data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        
        return combined_data

    def get_crypto_metadata(self) -> Dict:
        """
        Fetch metadata for selected cryptocurrencies
        
        Returns:
            Dict: Cryptocurrency metadata
        """
        metadata = {}
        
        for crypto in self.cryptocurrencies:
            try:
                url = f"{self.base_url}/coins/{crypto}"
                response = requests.get(url)
                response.raise_for_status()
                
                crypto_info = response.json()
                metadata[crypto] = {
                    'name': crypto_info.get('name'),
                    'symbol': crypto_info.get('symbol'),
                    'description': crypto_info.get('description', {}).get('en', ''),
                    'market_cap_rank': crypto_info.get('market_cap_rank'),
                    'total_volume': crypto_info.get('market_data', {}).get('total_volume', {}).get('usd')
                }
            except requests.RequestException as e:
                print(f"Error fetching metadata for {crypto}: {e}")
        
        return metadata

def main():
    collector = CryptoDataCollector()
    
    # Collect price data
    price_data = collector.collect_multi_crypto_data()
    
    # Collect metadata
    metadata = collector.get_crypto_metadata()
    print("Cryptocurrency Metadata:")
    for crypto, info in metadata.items():
        print(f"{crypto.capitalize()}: {info}")

if __name__ == "__main__":
    main()
```
