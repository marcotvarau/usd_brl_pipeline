"""
CSV Exporter Module
Exports pipeline data to CSV format with compression options
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import logging
import gzip
import zipfile


class CSVExporter:
    """
    Exports data to CSV format.
    
    Features:
    - Multiple compression formats (gzip, zip, bz2)
    - Chunked writing for large files
    - Data type optimization
    - Metadata generation
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize CSV exporter.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Output configuration
        output_config = config.get('output', {})
        self.base_path = Path(output_config.get('base_path', 'data'))
        self.features_dir = self.base_path / output_config.get('subdirs', {}).get('features', 'features')
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV configuration
        self.compression = output_config.get('compression', {}).get('csv', 'gzip')
        self.chunk_size = config.get('performance', {}).get('chunk_size', 10000)
        self.float_precision = 6
        self.date_format = '%Y-%m-%d'
        
        # File naming
        self.file_pattern = output_config.get('file_naming', {}).get('pattern', 'usd_brl_{tier}_{date}_{version}')
        self.date_pattern = output_config.get('file_naming', {}).get('date_format', '%Y%m%d')
    
    def export(
        self,
        data: pd.DataFrame,
        filename: Optional[str] = None,
        tier: str = 'all',
        compression: Optional[str] = None,
        include_metadata: bool = True,
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Export DataFrame to CSV.
        
        Args:
            data: DataFrame to export
            filename: Custom filename (without extension)
            tier: Data tier (tier1, tier2, tier3, tier4, all)
            compression: Compression type override
            include_metadata: Whether to generate metadata file
            chunk_size: Override chunk size for writing
            
        Returns:
            Export results dictionary
        """
        try:
            self.logger.info(f"Exporting data to CSV: {len(data)} rows, {len(data.columns)} columns")
            
            # Generate filename
            if filename is None:
                filename = self._generate_filename(tier)
            
            # Filter columns by tier
            filtered_data = self._filter_by_tier(data, tier)
            
            # Optimize data types
            optimized_data = self._optimize_datatypes(filtered_data)
            
            # Determine file path and compression
            compression = compression or self.compression
            file_path = self._get_file_path(filename, compression)
            
            # Export data
            if len(optimized_data) > (chunk_size or self.chunk_size):
                self._export_chunked(optimized_data, file_path, compression, chunk_size)
            else:
                self._export_single(optimized_data, file_path, compression)
            
            # Generate metadata
            metadata = None
            if include_metadata:
                metadata = self._generate_metadata(optimized_data, file_path)
                metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
                self._save_metadata(metadata, metadata_path)
            
            # Calculate file size
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            
            result = {
                'status': 'success',
                'path': str(file_path),
                'rows': len(optimized_data),
                'columns': len(optimized_data.columns),
                'file_size_mb': round(file_size, 2),
                'compression': compression,
                'metadata': metadata
            }
            
            self.logger.info(f"CSV export successful: {file_path} ({file_size:.2f} MB)")
            return result
            
        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def export_multiple(
        self,
        datasets: Dict[str, pd.DataFrame],
        prefix: str = None
    ) -> Dict[str, Any]:
        """
        Export multiple DataFrames to separate CSV files.
        
        Args:
            datasets: Dictionary of name: DataFrame pairs
            prefix: Filename prefix
            
        Returns:
            Export results for each dataset
        """
        results = {}
        
        for name, data in datasets.items():
            filename = f"{prefix}_{name}" if prefix else name
            result = self.export(data, filename=filename)
            results[name] = result
        
        return results
    
    def _generate_filename(self, tier: str) -> str:
        """Generate filename based on pattern."""
        date_str = datetime.now().strftime(self.date_pattern)
        version = self.config.get('version', '1.0.0').replace('.', '_')
        
        filename = self.file_pattern.format(
            tier=tier,
            date=date_str,
            version=version
        )
        
        return filename
    
    def _filter_by_tier(self, data: pd.DataFrame, tier: str) -> pd.DataFrame:
        """Filter columns by tier."""
        if tier == 'all':
            return data
        
        # Get tier configuration
        tier_config = self.config.get('feature_tiers', {}).get(tier, {})
        tier_features = tier_config.get('features', [])
        
        if not tier_features:
            self.logger.warning(f"No features configured for tier {tier}, returning all data")
            return data
        
        # Find matching columns
        matching_cols = []
        for feature_pattern in tier_features:
            matching = [col for col in data.columns if feature_pattern in col]
            matching_cols.extend(matching)
        
        # Always include the target variable
        if 'usd_brl_ptax_close' in data.columns and 'usd_brl_ptax_close' not in matching_cols:
            matching_cols.append('usd_brl_ptax_close')
        
        # Remove duplicates while preserving order
        matching_cols = list(dict.fromkeys(matching_cols))
        
        if not matching_cols:
            self.logger.warning(f"No columns match tier {tier} patterns")
            return data
        
        return data[matching_cols]
    
    def _optimize_datatypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce file size."""
        optimized = data.copy()
        
        for col in optimized.columns:
            col_type = str(optimized[col].dtype)
            
            # Optimize numeric types
            if 'float' in col_type:
                # Check if can be converted to int
                if optimized[col].notna().all() and (optimized[col] == optimized[col].astype(int)).all():
                    optimized[col] = optimized[col].astype(int)
                else:
                    # Use float32 if precision allows
                    if optimized[col].abs().max() < 1e6:
                        optimized[col] = optimized[col].astype('float32')
            
            elif 'int' in col_type:
                # Downcast integers
                if optimized[col].min() >= 0:
                    if optimized[col].max() < 256:
                        optimized[col] = optimized[col].astype('uint8')
                    elif optimized[col].max() < 65536:
                        optimized[col] = optimized[col].astype('uint16')
                else:
                    if optimized[col].min() > -128 and optimized[col].max() < 127:
                        optimized[col] = optimized[col].astype('int8')
                    elif optimized[col].min() > -32768 and optimized[col].max() < 32767:
                        optimized[col] = optimized[col].astype('int16')
            
            # Optimize boolean columns
            elif col_type == 'bool' or (optimized[col].dropna().isin([0, 1]).all()):
                optimized[col] = optimized[col].astype(bool)
        
        return optimized
    
    def _get_file_path(self, filename: str, compression: str) -> Path:
        """Get full file path with appropriate extension."""
        base_name = filename if not filename.endswith('.csv') else filename[:-4]
        
        if compression == 'gzip':
            file_path = self.features_dir / f"{base_name}.csv.gz"
        elif compression == 'zip':
            file_path = self.features_dir / f"{base_name}.csv.zip"
        elif compression == 'bz2':
            file_path = self.features_dir / f"{base_name}.csv.bz2"
        elif compression == 'xz':
            file_path = self.features_dir / f"{base_name}.csv.xz"
        else:
            file_path = self.features_dir / f"{base_name}.csv"
        
        return file_path
    
    def _export_single(
        self,
        data: pd.DataFrame,
        file_path: Path,
        compression: str
    ):
        """Export data in a single write."""
        compression_map = {
            'gzip': 'gzip',
            'bz2': 'bz2',
            'xz': 'xz',
            'zip': None,  # Handle separately
            'none': None
        }
        
        if compression == 'zip':
            # Special handling for zip
            zip_path = file_path
            csv_name = file_path.stem  # Remove .zip
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                csv_data = data.to_csv(
                    index=True,
                    float_format=f'%.{self.float_precision}f',
                    date_format=self.date_format
                )
                zf.writestr(csv_name, csv_data)
        else:
            # Standard pandas compression
            data.to_csv(
                file_path,
                index=True,
                compression=compression_map.get(compression),
                float_format=f'%.{self.float_precision}f',
                date_format=self.date_format
            )
    
    def _export_chunked(
        self,
        data: pd.DataFrame,
        file_path: Path,
        compression: str,
        chunk_size: Optional[int] = None
    ):
        """Export data in chunks for memory efficiency."""
        chunk_size = chunk_size or self.chunk_size
        
        # For compressed files, we need to write all at once
        if compression != 'none':
            self._export_single(data, file_path, compression)
            return
        
        # For uncompressed, we can append chunks
        for i, start_idx in enumerate(range(0, len(data), chunk_size)):
            end_idx = min(start_idx + chunk_size, len(data))
            chunk = data.iloc[start_idx:end_idx]
            
            mode = 'w' if i == 0 else 'a'
            header = i == 0
            
            chunk.to_csv(
                file_path,
                mode=mode,
                header=header,
                index=True,
                float_format=f'%.{self.float_precision}f',
                date_format=self.date_format
            )
    
    def _generate_metadata(
        self,
        data: pd.DataFrame,
        file_path: Path
    ) -> Dict[str, Any]:
        """Generate metadata for exported file."""
        # Basic metadata
        metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'export_timestamp': datetime.now().isoformat(),
            'pipeline_version': self.config.get('version', '1.0.0'),
            
            # Data characteristics
            'shape': {
                'rows': len(data),
                'columns': len(data.columns)
            },
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'file_size_mb': file_path.stat().st_size / 1024 / 1024,
            
            # Date range
            'date_range': {
                'start': str(data.index.min()),
                'end': str(data.index.max()),
                'days': (data.index.max() - data.index.min()).days if isinstance(data.index, pd.DatetimeIndex) else None
            },
            
            # Column information
            'columns': {
                'names': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'missing_values': data.isnull().sum().to_dict(),
                'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict()
            },
            
            # Statistical summary
            'statistics': {}
        }
        
        # Add statistical summary for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:20]:  # Limit to first 20 columns
            if col in data.columns:
                metadata['statistics'][col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'median': float(data[col].median()),
                    'q25': float(data[col].quantile(0.25)),
                    'q75': float(data[col].quantile(0.75))
                }
        
        return metadata
    
    def _save_metadata(self, metadata: Dict[str, Any], path: Path):
        """Save metadata to JSON file."""
        import json
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.debug(f"Metadata saved to {path}")