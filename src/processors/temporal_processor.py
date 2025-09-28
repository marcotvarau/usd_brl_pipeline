"""
Temporal Features Processor
Creates time-based features like lags, rolling windows, and seasonal patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging


class TemporalProcessor:
    """
    Processor for creating temporal features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the temporal processor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get feature configuration
        features_config = config.get('features', {})
        self.lag_periods = features_config.get('lag_periods', [1, 2, 5, 10, 22])
        self.rolling_windows = features_config.get('rolling_windows', [5, 10, 22])
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features.
        
        Args:
            data: Input DataFrame with time series data
            
        Returns:
            DataFrame with temporal features added
        """
        self.logger.info("Creating temporal features")
        
        result = data.copy()
        
        # Create lag features
        result = self._create_lag_features(result)
        
        # Create rolling window features
        result = self._create_rolling_features(result)
        
        # Create seasonal features
        result = self._create_seasonal_features(result)
        
        return result
    
    def _create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features."""
        result = data.copy()
        
        # Key columns to create lags for
        key_columns = ['usd_brl_ptax_close', 'selic_rate', 'fed_funds_rate', 'dxy_index']
        
        for col in key_columns:
            if col in result.columns:
                for lag in self.lag_periods:
                    result[f'{col}_lag_{lag}'] = result[col].shift(lag)
        
        return result
    
    def _create_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features."""
        result = data.copy()
        
        # Key columns for rolling features
        key_columns = ['usd_brl_ptax_close', 'selic_rate', 'fed_funds_rate']
        
        for col in key_columns:
            if col in result.columns:
                for window in self.rolling_windows:
                    result[f'{col}_rolling_mean_{window}'] = result[col].rolling(window).mean()
                    result[f'{col}_rolling_std_{window}'] = result[col].rolling(window).std()
        
        return result
    
    def _create_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal/calendar features."""
        result = data.copy()
        
        # Ensure index is datetime
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)
        
        # Calendar features
        result['day_of_week'] = result.index.dayofweek
        result['day_of_month'] = result.index.day
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        result['is_month_end'] = result.index.is_month_end.astype(int)
        result['is_quarter_end'] = result.index.is_quarter_end.astype(int)
        
        # Cyclical encoding
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        return result
