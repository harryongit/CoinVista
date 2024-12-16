import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preprocessor import CryptoDataPreprocessor

class TestCryptoDataPreprocessor:
    @pytest.fixture
    def sample_data(self):
        """Create a sample cryptocurrency DataFrame for testing"""
        dates = pd.date_range(start='2023-01-01', periods=30)
        return pd.DataFrame({
            'timestamp': dates,
            'crypto': ['bitcoin'] * 30,
            'price': np.linspace(40000, 45000, 30)
        })

    @pytest.fixture
    def preprocessor(self, sample_data):
        """Create a CryptoDataPreprocessor instance with sample data"""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            sample_data.to_csv(temp_file.name, index=False)
        
        # Create preprocessor with the temporary file
        preprocessor = CryptoDataPreprocessor(data_path=temp_file.name)
        
        return preprocessor

    def test_init(self, preprocessor):
        """Test initialization of CryptoDataPreprocessor"""
        assert isinstance(preprocessor.data, pd.DataFrame)
        assert not preprocessor.data.empty
        assert os.path.exists(preprocessor.processed_data_dir)

    def test_preprocess_data(self, preprocessor):
        """Test data preprocessing method"""
        processed_data = preprocessor.preprocess_data()

        # Assertions
        assert isinstance(processed_data, pd.DataFrame)
        assert 'timestamp' in processed_data.columns
        assert 'crypto' in processed_data.columns
        assert 'price' in processed_data.columns

    def test_calculate_features(self, preprocessor):
        """Test feature calculation method"""
        preprocessed_data = preprocessor.preprocess_data()
        featured_data = preprocessor.calculate_features(preprocessed_data)

        # Assertions
        assert 'rolling_mean_7d' in featured_data.columns
        assert 'rolling_std_7d' in featured_data.columns
        assert 'daily_return' in featured_data.columns
        assert 'cumulative_return' in featured_data.columns

    def test_detect_outliers(self, preprocessor):
        """Test outlier detection method"""
        preprocessed_data = preprocessor.preprocess_data()
        featured_data = preprocessor.calculate_features(preprocessed_data)
        outliers = preprocessor.detect_outliers(featured_data)

        # Assertions
        assert isinstance(outliers, dict)
        assert len(outliers) > 0
        for crypto, crypto_outliers in outliers.items():
            assert isinstance(crypto_outliers, pd.DataFrame)

    def test_save_processed_data(self, preprocessor):
        """Test saving processed data"""
        preprocessed_data = preprocessor.preprocess_data()
        featured_data = preprocessor.calculate_features(preprocessed_data)

        # Save the data
        preprocessor.save_processed_data(featured_data)

        # Check if file was created
        processed_files = os.listdir(preprocessor.processed_data_dir)
        assert any(f.startswith('processed_crypto_data') and f.endswith('.csv') 
                   for f in processed_files)

    def test_visualize_preprocessing(self, preprocessor):
        """Test preprocessing visualization"""
        preprocessed_data = preprocessor.preprocess_data()
        featured_data = preprocessor.calculate_features(preprocessed_data)

        # Call visualization method
        preprocessor.visualize_preprocessing(featured_data)

        # Check if visualization was saved
        viz_file = os.path.join(preprocessor.processed_data_dir, 'preprocessing_visualization.png')
        assert os.path.exists(viz_file)

    def test_edge_cases(self, preprocessor):
        """Test edge cases and error handling"""
        # Test with empty or invalid data
        with pytest.raises(Exception):
            invalid_data = pd.DataFrame()
            preprocessor.calculate_features(invalid_data)
