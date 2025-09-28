"""
Data Cleaner Processor
Handles data cleaning, outlier detection, and missing value imputation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from scipy import stats


class DataCleaner:
    """
    Processor for cleaning and preprocessing data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data cleaner."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get validation configuration
        validation_config = config.get('validation', {})
        self.ranges = validation_config.get('ranges', {})
        self.max_missing_pct = validation_config.get('max_missing_pct', 5.0)
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input data.
        
        Args:
            data: Input DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning process")
        
        result = data.copy()
        
        # Handle missing values
        result = self._handle_missing_values(result)
        
        # Remove outliers
        result = self._handle_outliers(result)
        
        # Validate ranges
        result = self._validate_ranges(result)
        
        # Remove duplicates
        result = self._remove_duplicates(result)
        
        self.logger.info(f"Data cleaning complete. Shape: {result.shape}")
        
        return result
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        result = data.copy()
        
        # Check missing percentage
        missing_pct = (result.isnull().sum() / len(result)) * 100
        
        for col, pct in missing_pct.items():
            if pct > 0:
                self.logger.info(f"Column {col}: {pct:.2f}% missing")
                
                if pct > self.max_missing_pct:
                    self.logger.warning(f"Column {col} has {pct:.2f}% missing (> {self.max_missing_pct}%)")
        
        # Forward fill then backward fill for time series
        result = result.fillna(method='ffill').fillna(method='bfill')
        
        # For remaining NaNs, use interpolation
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        result[numeric_columns] = result[numeric_columns].interpolate()
        
        return result
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        result = data.copy()
        
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in result.columns:
                # Calculate IQR
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = ((result[col] < lower_bound) | (result[col] > upper_bound)).sum()
                
                if outliers > 0:
                    self.logger.info(f"Column {col}: {outliers} outliers detected")
                    
                    # Cap outliers instead of removing
                    result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        
        return result
    
    def _validate_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges based on configuration."""
        result = data.copy()
        
        for col, range_config in self.ranges.items():
            if col in result.columns:
                min_val = range_config.get('min')
                max_val = range_config.get('max')
                
                if min_val is not None:
                    invalid_min = (result[col] < min_val).sum()
                    if invalid_min > 0:
                        self.logger.warning(f"Column {col}: {invalid_min} values below minimum {min_val}")
                        result[col] = result[col].clip(lower=min_val)
                
                if max_val is not None:
                    invalid_max = (result[col] > max_val).sum()
                    if invalid_max > 0:
                        self.logger.warning(f"Column {col}: {invalid_max} values above maximum {max_val}")
                        result[col] = result[col].clip(upper=max_val)
        
        return result
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        result = data.copy()
        
        initial_rows = len(result)
        result = result.drop_duplicates()
        final_rows = len(result)
        
        duplicates_removed = initial_rows - final_rows
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return result
