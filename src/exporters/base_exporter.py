"""
Base Exporter Abstract Class
Defines the interface for all data exporters in the pipeline
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging


class BaseExporter(ABC):
    """
    Abstract base class for all data exporters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def export(self, data: pd.DataFrame, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        Export data to specified format.
        
        Args:
            data: DataFrame to export
            output_path: Output file path
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with export results
        """
        pass
