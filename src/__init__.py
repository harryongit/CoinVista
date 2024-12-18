"""
Cryptocurrency Time Series Analysis Project
Main Package Initialization
"""

# Import core modules
from .data_collector import CryptoDataCollector
from .data_preprocessor import CryptoDataPreprocessor
from .time_series_analyzer import CryptoTimeSeriesAnalyzer
from .machine_learning_models import CryptoPricePredictionModels

# Version information
__version__ = "0.1.0"
__author__ = "Harivdan N"
__email__ = "harryshastri21@gmail.com"

# Package-level configurations
CONFIG = {
    "supported_cryptocurrencies": [
        "bitcoin", 
        "ethereum", 
        "binancecoin", 
        "cardano", 
        "solana"
    ],
    "default_lookback_days": 365,
    "data_directories": {
        "raw": "data/raw",
        "processed": "data/processed",
        "models": "models",
        "outputs": "outputs"
    }
}

# Expose key classes for easy importing
__all__ = [
    "CryptoDataCollector",
    "CryptoDataPreprocessor", 
    "CryptoTimeSeriesAnalyzer",
    "CryptoPricePredictionModels"
]

# Basic logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
