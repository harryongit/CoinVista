import pytest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_collector import CryptoDataCollector

class TestCryptoDataCollector:
    @pytest.fixture
    def data_collector(self):
        """Create a CryptoDataCollector instance for testing"""
        return CryptoDataCollector()

    def test_init(self, data_collector):
        """Test initialization of CryptoDataCollector"""
        assert isinstance(data_collector.cryptocurrencies, list)
        assert len(data_collector.cryptocurrencies) > 0
        assert data_collector.days == 365
        assert os.path.exists(data_collector.data_dir)

    @patch('requests.get')
    def test_fetch_historical_data(self, mock_get, data_collector):
        """Test fetching historical data for a cryptocurrency"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'prices': [
                [1622505600000, 40000.0],
                [1622592000000, 41000.0]
            ]
        }
        mock_get.return_value = mock_response

        # Fetch data
        result = data_collector.fetch_historical_data('bitcoin')

        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'timestamp' in result.columns
        assert 'price' in result.columns
        assert 'crypto' in result.columns

    @patch('requests.get')
    def test_get_crypto_metadata(self, mock_get, data_collector):
        """Test fetching cryptocurrency metadata"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'name': 'Bitcoin',
            'symbol': 'BTC',
            'description': {'en': 'A decentralized digital currency'},
            'market_cap_rank': 1,
            'market_data': {
                'total_volume': {'usd': 30000000000}
            }
        }
        mock_get.return_value = mock_response

        # Fetch metadata
        metadata = data_collector.get_crypto_metadata()

        # Assertions
        assert isinstance(metadata, dict)
        assert len(metadata) > 0
        
        for crypto, info in metadata.items():
            assert 'name' in info
            assert 'symbol' in info
            assert 'description' in info
            assert 'market_cap_rank' in info
            assert 'total_volume' in info

    def test_collect_multi_crypto_data(self, data_collector):
        """Test collecting data for multiple cryptocurrencies"""
        # Mock the fetch_historical_data method
        with patch.object(data_collector, 'fetch_historical_data', 
                          return_value=pd.DataFrame({
                              'timestamp': pd.date_range(start='2023-01-01', periods=10),
                              'price': [100 + i for i in range(10)],
                              'crypto': ['bitcoin'] * 10
                          })):
            
            result = data_collector.collect_multi_crypto_data()

            # Assertions
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'timestamp' in result.columns
            assert 'price' in result.columns
            assert 'crypto' in result.columns

    def test_rate_limiting(self, data_collector):
        """Test that there's a pause between API calls"""
        import time

        with patch('requests.get'), patch('time.sleep') as mock_sleep:
            data_collector.collect_multi_crypto_data()

            # Verify that time.sleep was called
            assert mock_sleep.call_count >= len(data_collector.cryptocurrencies) - 1

    def test_error_handling(self, data_collector):
        """Test error handling during data collection"""
        # Simulate a network error
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")

            result = data_collector.fetch_historical_data('bitcoin')

            # Should return an empty DataFrame
            assert isinstance(result, pd.DataFrame)
            assert result.empty
