"""
Data Drift Monitor
Monitors data drift and distribution changes over time
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from scipy import stats


class DriftMonitor:
    """
    Monitor for data drift detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the drift monitor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get monitoring configuration
        monitoring_config = config.get('monitoring', {})
        alerts_config = monitoring_config.get('alerts', {})
        self.drift_threshold = alerts_config.get('data_drift_threshold', 0.2)
    
    def monitor(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Monitor data for drift.
        
        Args:
            data: Input DataFrame to monitor
            
        Returns:
            Dictionary with drift monitoring results
        """
        self.logger.info("Running drift monitoring")
        
        result = {
            'status': 'passed',
            'drift_detected': False,
            'metrics': {},
            'issues': []
        }
        
        # Basic drift detection using statistical tests
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in data.columns and len(data[col].dropna()) > 10:
                # Simple drift detection using rolling statistics
                drift_score = self._calculate_drift_score(data[col])
                result['metrics'][f'{col}_drift_score'] = drift_score
                
                if drift_score > self.drift_threshold:
                    result['drift_detected'] = True
                    result['status'] = 'warning'
                    result['issues'].append(f'Drift detected in {col}: score {drift_score:.3f}')
        
        return result
    
    def _calculate_drift_score(self, series: pd.Series) -> float:
        """
        Calculate drift score for a series.
        Simple implementation using coefficient of variation changes.
        """
        if len(series) < 20:
            return 0.0
        
        # Split series into two halves
        mid_point = len(series) // 2
        first_half = series.iloc[:mid_point]
        second_half = series.iloc[mid_point:]
        
        # Calculate coefficient of variation for each half
        cv1 = first_half.std() / abs(first_half.mean()) if first_half.mean() != 0 else 0
        cv2 = second_half.std() / abs(second_half.mean()) if second_half.mean() != 0 else 0
        
        # Drift score as absolute difference in CV
        drift_score = abs(cv2 - cv1)
        
        return drift_score
