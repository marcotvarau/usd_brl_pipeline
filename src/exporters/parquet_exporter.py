"""
Parquet Exporter
Exports data to Parquet format
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path
from datetime import datetime

from .base_exporter import BaseExporter


class ParquetExporter(BaseExporter):
    """
    Exporter for Parquet format.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the parquet exporter."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Get compression setting
        compression_config = config.get('output', {}).get('compression', {})
        self.compression = compression_config.get('parquet', 'snappy')
    
    def export(self, data: pd.DataFrame, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        Export data to Parquet format.
        
        Args:
            data: DataFrame to export
            output_path: Output file path
            **kwargs: Additional parameters
            
        Returns:
            Export result dictionary
        """
        try:
            self.logger.info(f"Exporting data to Parquet: {output_path}")
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export to parquet
            data.to_parquet(
                output_path,
                compression=self.compression,
                index=True
            )
            
            # Get file info
            file_path = Path(output_path)
            file_size = file_path.stat().st_size if file_path.exists() else 0
            
            result = {
                'status': 'success',
                'file_path': str(file_path.absolute()),
                'file_size_bytes': file_size,
                'compression': self.compression,
                'rows_exported': len(data),
                'columns_exported': len(data.columns)
            }
            
            self.logger.info(f"Successfully exported {len(data)} rows to {output_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to export to Parquet: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'file_path': output_path
            }
