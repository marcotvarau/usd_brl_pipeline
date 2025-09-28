"""
Data Integrity Checker
Validates data integrity and consistency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging


class IntegrityChecker:
    """
    Checker for data integrity validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the integrity checker."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get validation configuration
        validation_config = config.get('validation', {})
        self.required_columns = validation_config.get('required_columns', [])
        self.expected_correlations = validation_config.get('expected_correlations', [])
    
    def check(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data integrity.
        
        Args:
            data: Input DataFrame to check
            
        Returns:
            Dictionary with integrity check results
        """
        self.logger.info("Running integrity checks")
        
        result = {
            'status': 'passed',
            'passed': True,
            'metrics': {},
            'issues': []
        }
        
        # Check required columns
        result = self._check_required_columns(data, result)
        
        # Check correlations
        result = self._check_correlations(data, result)
        
        # Check data consistency
        result = self._check_consistency(data, result)
        
        if result['issues']:
            result['status'] = 'failed'
            result['passed'] = False
        
        return result
    
    def _check_required_columns(self, data: pd.DataFrame, result: Dict[str, Any]) -> Dict[str, Any]:
        """Check if required columns are present."""
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        
        if missing_columns:
            result['issues'].append(f'Missing required columns: {missing_columns}')
        
        result['metrics']['required_columns_present'] = len(self.required_columns) - len(missing_columns)
        result['metrics']['required_columns_total'] = len(self.required_columns)
        
        return result
    
    def _check_correlations(self, data: pd.DataFrame, result: Dict[str, Any]) -> Dict[str, Any]:
        """Check expected correlations."""
        for corr_check in self.expected_correlations:
            columns = corr_check.get('columns', [])
            expected_range = corr_check.get('range', [-1, 1])
            
            if len(columns) == 2 and all(col in data.columns for col in columns):
                correlation = data[columns[0]].corr(data[columns[1]])
                
                if not (expected_range[0] <= correlation <= expected_range[1]):
                    result['issues'].append(
                        f'Correlation between {columns[0]} and {columns[1]} is {correlation:.3f}, '
                        f'expected range: {expected_range}'
                    )
                
                result['metrics'][f'correlation_{columns[0]}_{columns[1]}'] = correlation
        
        return result
    
    def _check_consistency(self, data: pd.DataFrame, result: Dict[str, Any]) -> Dict[str, Any]:
        """Check data consistency."""
        # Check for monotonic time index
        if isinstance(data.index, pd.DatetimeIndex):
            if not data.index.is_monotonic_increasing:
                result['issues'].append('Time index is not monotonic increasing')
        
        # Check for reasonable data ranges
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if len(data[col].dropna()) > 0:
                # Check for extreme values (more than 5 standard deviations)
                mean_val = data[col].mean()
                std_val = data[col].std()
                extreme_values = ((data[col] - mean_val).abs() > 5 * std_val).sum()
                
                if extreme_values > 0:
                    result['issues'].append(f'Column {col} has {extreme_values} extreme values')
                
                result['metrics'][f'{col}_extreme_values'] = extreme_values
        
        return result
