"""
Main Pipeline Orchestrator
Coordinates the entire USD/BRL data pipeline from collection to export
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import yaml
import os
from pathlib import Path
import concurrent.futures
import traceback
import json
import time

# Import pipeline components
from ..collectors.bcb_collector import BCBCollector
from ..collectors.fred_collector import FREDCollector
from ..collectors.yahoo_collector import YahooCollector
from ..collectors.cot_collector import COTCollector
from ..collectors.news_collector import NewsCollector

from ..processors.feature_engineer import FeatureEngineer
from ..processors.technical_indicators import TechnicalIndicatorProcessor
from ..processors.temporal_processor import TemporalProcessor
from ..processors.data_cleaner import DataCleaner

from ..validators.quality_validator import QualityValidator
from ..validators.drift_monitor import DriftMonitor
from ..validators.integrity_checker import IntegrityChecker

from ..exporters.csv_exporter import CSVExporter
from ..exporters.parquet_exporter import ParquetExporter
from ..exporters.database_exporter import DatabaseExporter

from ..utils.logger import setup_logger
from ..utils.cache_manager import CacheManager
from ..utils.circuit_breaker import CircuitBreaker


class PipelineOrchestrator:
    """
    Main orchestrator that coordinates the entire data pipeline.
    
    Responsibilities:
    - Initialize all components
    - Coordinate data collection from multiple sources
    - Manage parallel processing
    - Handle errors and retries
    - Monitor pipeline performance
    - Export data in multiple formats
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self.logger = setup_logger(
            'PipelineOrchestrator',
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file='logs/pipeline.log'
        )
        
        self.logger.info("Initializing Pipeline Orchestrator")
        
        # Initialize components
        self._initialize_components()
        
        # Pipeline state
        self.pipeline_state = {
            'status': 'initialized',
            'start_time': None,
            'end_time': None,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Data storage
        self.raw_data = {}
        self.processed_data = None
        self.validated_data = None
        self.feature_data = None
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Substitute environment variables
        config = self._substitute_env_vars(config)
        
        return config
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in config.
        
        Args:
            config: Configuration object
            
        Returns:
            Config with environment variables substituted
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        
        # Initialize collectors
        self.collectors = {
            'bcb': BCBCollector(self.config, logger=self.logger),
            'fred': FREDCollector(self.config, logger=self.logger),
            'yahoo': YahooCollector(self.config, logger=self.logger),
            'cot': COTCollector(self.config, logger=self.logger),
            'news': NewsCollector(self.config, logger=self.logger)
        }
        
        # Initialize processors
        self.processors = {
            'feature_engineer': FeatureEngineer(self.config, logger=self.logger),
            'technical': TechnicalIndicatorProcessor(self.config, logger=self.logger),
            'temporal': TemporalProcessor(self.config, logger=self.logger),
            'cleaner': DataCleaner(self.config, logger=self.logger)
        }
        
        # Initialize validators
        self.validators = {
            'quality': QualityValidator(self.config, logger=self.logger),
            'drift': DriftMonitor(self.config, logger=self.logger),
            'integrity': IntegrityChecker(self.config, logger=self.logger)
        }
        
        # Initialize exporters
        self.exporters = {
            'csv': CSVExporter(self.config, logger=self.logger),
            'parquet': ParquetExporter(self.config, logger=self.logger),
            'database': DatabaseExporter(self.config, logger=self.logger)
        }
        
        # Initialize utilities
        self.cache_manager = CacheManager(self.config, logger=self.logger)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get('circuit_breaker', {}).get('failure_threshold', 5),
            recovery_timeout=self.config.get('circuit_breaker', {}).get('recovery_timeout', 60)
        )
        
        self.logger.info("All components initialized successfully")
    
    def run_pipeline(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        collectors: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete data pipeline.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            collectors: List of collectors to use (None = all)
            force_refresh: Force refresh ignoring cache
            
        Returns:
            Dictionary with pipeline results and metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting USD/BRL Data Pipeline Run")
        self.logger.info("=" * 80)
        
        # Update pipeline state
        self.pipeline_state['status'] = 'running'
        self.pipeline_state['start_time'] = datetime.now()
        self.pipeline_state['errors'] = []
        self.pipeline_state['warnings'] = []
        
        try:
            # 1. Setup dates
            if start_date is None:
                start_date = self.config.get('collection', {}).get('start_date', '2014-01-01')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            self.logger.info(f"Date range: {start_date} to {end_date}")
            
            # 2. Collect data
            self.logger.info("Phase 1: Data Collection")
            self.raw_data = self._collect_data(start_date, end_date, collectors, force_refresh)
            
            if not self.raw_data:
                raise ValueError("No data collected")
            
            # 3. Merge and clean data
            self.logger.info("Phase 2: Data Merging and Cleaning")
            self.processed_data = self._process_data(self.raw_data)
            
            # 4. Feature engineering
            self.logger.info("Phase 3: Feature Engineering")
            self.feature_data = self._engineer_features(self.processed_data)
            
            # 5. Validation
            self.logger.info("Phase 4: Data Validation")
            validation_results = self._validate_data(self.feature_data)
            
            if not validation_results['passed']:
                self.pipeline_state['warnings'].extend(validation_results['issues'])
                self.logger.warning(f"Validation issues found: {validation_results['issues']}")
            
            self.validated_data = self.feature_data
            
            # 6. Export data
            self.logger.info("Phase 5: Data Export")
            export_results = self._export_data(self.validated_data)
            
            # 7. Generate metrics
            self.pipeline_state['metrics'] = self._generate_metrics()
            
            # Update pipeline state
            self.pipeline_state['status'] = 'completed'
            self.pipeline_state['end_time'] = datetime.now()
            
            # Generate summary
            summary = self._generate_summary()
            
            self.logger.info("=" * 80)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Total time: {self.pipeline_state['end_time'] - self.pipeline_state['start_time']}")
            self.logger.info("=" * 80)
            
            return {
                'status': 'success',
                'data': self.validated_data,
                'summary': summary,
                'metrics': self.pipeline_state['metrics'],
                'export_results': export_results
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            self.pipeline_state['status'] = 'failed'
            self.pipeline_state['end_time'] = datetime.now()
            self.pipeline_state['errors'].append(str(e))
            
            return {
                'status': 'failed',
                'error': str(e),
                'state': self.pipeline_state
            }
    
    def _collect_data(
        self,
        start_date: str,
        end_date: str,
        collectors: Optional[List[str]],
        force_refresh: bool
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all sources.
        
        Args:
            start_date: Start date
            end_date: End date
            collectors: List of collectors to use
            force_refresh: Force refresh
            
        Returns:
            Dictionary of collected data
        """
        if collectors is None:
            collectors = list(self.collectors.keys())
        
        collected_data = {}
        
        # Use thread pool for parallel collection
        max_workers = self.config.get('performance', {}).get('n_workers', 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit collection tasks
            future_to_collector = {}
            
            for name in collectors:
                if name in self.collectors:
                    collector = self.collectors[name]
                    
                    # Check circuit breaker
                    if self.circuit_breaker.can_execute(name):
                        future = executor.submit(
                            self._collect_with_circuit_breaker,
                            collector,
                            start_date,
                            end_date,
                            force_refresh,
                            name
                        )
                        future_to_collector[future] = name
                    else:
                        self.logger.warning(f"Circuit breaker open for {name}, skipping")
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_collector):
                collector_name = future_to_collector[future]
                
                try:
                    data = future.result(timeout=60)
                    if data is not None and not data.empty:
                        collected_data[collector_name] = data
                        self.logger.info(f"✓ Collected {len(data)} rows from {collector_name}")
                    else:
                        self.logger.warning(f"✗ No data from {collector_name}")
                        
                except Exception as e:
                    self.logger.error(f"✗ Failed to collect from {collector_name}: {e}")
                    self.pipeline_state['errors'].append(f"{collector_name}: {str(e)}")
                    self.circuit_breaker.record_failure(collector_name)
        
        return collected_data
    
    def _collect_with_circuit_breaker(
        self,
        collector,
        start_date: str,
        end_date: str,
        force_refresh: bool,
        name: str
    ) -> Optional[pd.DataFrame]:
        """
        Collect data with circuit breaker protection.
        
        Args:
            collector: Collector instance
            start_date: Start date
            end_date: End date
            force_refresh: Force refresh
            name: Collector name
            
        Returns:
            Collected DataFrame or None
        """
        try:
            # Check cache if not forcing refresh
            if not force_refresh:
                cache_key = f"{name}_{start_date}_{end_date}"
                cached_data = self.cache_manager.get(cache_key)
                
                if cached_data is not None:
                    self.logger.info(f"Using cached data for {name}")
                    return cached_data
            
            # Collect data
            data = collector.collect(start_date, end_date)
            
            # Cache successful collection
            if data is not None and not data.empty:
                cache_key = f"{name}_{start_date}_{end_date}"
                self.cache_manager.set(cache_key, data, ttl=3600)
                self.circuit_breaker.record_success(name)
            
            return data
            
        except Exception as e:
            self.circuit_breaker.record_failure(name)
            raise
    
    def _process_data(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process and merge collected data.
        
        Args:
            raw_data: Dictionary of raw DataFrames
            
        Returns:
            Processed DataFrame
        """
        # Start with empty DataFrame
        processed = pd.DataFrame()
        
        # Process each data source
        for source, data in raw_data.items():
            self.logger.info(f"Processing {source} data: {data.shape}")
            
            # Clean data
            data = self.processors['cleaner'].clean(data)
            
            # Merge with main DataFrame
            if processed.empty:
                processed = data
            else:
                # Merge on index (date)
                processed = processed.join(data, how='outer')
        
        # Handle different frequencies
        processed = self.processors['cleaner'].align_frequencies(processed)
        
        # Final cleaning
        processed = self.processors['cleaner'].handle_missing_values(processed)
        
        self.logger.info(f"Processed data shape: {processed.shape}")
        
        return processed
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering.
        
        Args:
            data: Processed DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Apply feature engineering
        features = self.processors['feature_engineer'].engineer_all_features(data)
        
        # Add technical indicators
        features = self.processors['technical'].calculate_all_indicators(features)
        
        # Add temporal features
        features = self.processors['temporal'].create_all_temporal_features(features)
        
        self.logger.info(f"Feature data shape: {features.shape}")
        
        # Log feature statistics
        self._log_feature_statistics(features)
        
        return features
    
    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Validation results
        """
        results = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        # Quality validation
        quality_results = self.validators['quality'].validate(data)
        if not quality_results['passed']:
            results['passed'] = False
            results['issues'].extend(quality_results['issues'])
        results['metrics']['quality'] = quality_results['metrics']
        
        # Integrity check
        integrity_results = self.validators['integrity'].check(data)
        if not integrity_results['passed']:
            results['passed'] = False
            results['issues'].extend(integrity_results['issues'])
        results['metrics']['integrity'] = integrity_results['metrics']
        
        # Drift detection (if baseline exists)
        if self.validators['drift'].has_baseline():
            drift_results = self.validators['drift'].detect_drift(data)
            if drift_results['drift_detected']:
                results['issues'].append(f"Data drift detected: {drift_results['details']}")
            results['metrics']['drift'] = drift_results
        
        return results
    
    def _export_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Export data to configured formats.
        
        Args:
            data: DataFrame to export
            
        Returns:
            Export results
        """
        export_results = {}
        
        # Get configured formats
        formats = self.config.get('output', {}).get('formats', ['csv', 'parquet'])
        
        for format_name in formats:
            if format_name in self.exporters:
                try:
                    # Generate filename
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"usd_brl_pipeline_{timestamp}"
                    
                    # Export data
                    result = self.exporters[format_name].export(data, filename)
                    export_results[format_name] = result
                    
                    self.logger.info(f"✓ Exported to {format_name}: {result['path']}")
                    
                except Exception as e:
                    self.logger.error(f"✗ Failed to export to {format_name}: {e}")
                    export_results[format_name] = {'status': 'failed', 'error': str(e)}
        
        # Export to database if configured
        if self.config.get('database', {}).get('main', {}).get('enabled', False):
            try:
                db_result = self.exporters['database'].export(data, 'usd_brl_features')
                export_results['database'] = db_result
                self.logger.info(f"✓ Exported to database")
                
            except Exception as e:
                self.logger.error(f"✗ Failed to export to database: {e}")
                export_results['database'] = {'status': 'failed', 'error': str(e)}
        
        return export_results
    
    def _generate_metrics(self) -> Dict[str, Any]:
        """
        Generate pipeline metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'execution_time': str(self.pipeline_state['end_time'] - self.pipeline_state['start_time']),
            'data_points_collected': sum(len(df) for df in self.raw_data.values()),
            'features_created': len(self.feature_data.columns) if self.feature_data is not None else 0,
            'missing_data_percentage': 0,
            'collector_metrics': {},
            'processor_metrics': {},
            'memory_usage_mb': 0
        }
        
        # Calculate missing data percentage
        if self.feature_data is not None:
            metrics['missing_data_percentage'] = (
                self.feature_data.isnull().sum().sum() / 
                (len(self.feature_data) * len(self.feature_data.columns)) * 100
            )
        
        # Collector metrics
        for name, collector in self.collectors.items():
            metrics['collector_metrics'][name] = collector.get_metrics()
        
        # Memory usage
        if self.feature_data is not None:
            metrics['memory_usage_mb'] = self.feature_data.memory_usage(deep=True).sum() / 1024 / 1024
        
        return metrics
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate pipeline execution summary.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'pipeline_version': '1.0.0',
            'execution_date': datetime.now().isoformat(),
            'status': self.pipeline_state['status'],
            'data_sources': list(self.raw_data.keys()),
            'date_range': {
                'start': self.feature_data.index.min().isoformat() if self.feature_data is not None else None,
                'end': self.feature_data.index.max().isoformat() if self.feature_data is not None else None,
                'days': len(self.feature_data) if self.feature_data is not None else 0
            },
            'features': {
                'total': len(self.feature_data.columns) if self.feature_data is not None else 0,
                'tier1': len([c for c in self.feature_data.columns if 'tier1' in str(c).lower()]) if self.feature_data is not None else 0,
                'tier2': len([c for c in self.feature_data.columns if 'tier2' in str(c).lower()]) if self.feature_data is not None else 0,
                'tier3': len([c for c in self.feature_data.columns if 'tier3' in str(c).lower()]) if self.feature_data is not None else 0,
                'tier4': len([c for c in self.feature_data.columns if 'tier4' in str(c).lower()]) if self.feature_data is not None else 0
            },
            'quality_metrics': {
                'completeness': 100 - self.pipeline_state['metrics'].get('missing_data_percentage', 0),
                'errors': len(self.pipeline_state['errors']),
                'warnings': len(self.pipeline_state['warnings'])
            }
        }
        
        return summary
    
    def _log_feature_statistics(self, data: pd.DataFrame):
        """
        Log feature statistics for monitoring.
        
        Args:
            data: Feature DataFrame
        """
        # Key features to monitor
        key_features = [
            'usd_brl_ptax_close',
            'real_interest_differential',
            'risk_sentiment_score',
            'brazilian_commodity_index'
        ]
        
        stats = []
        for feature in key_features:
            if feature in data.columns:
                stats.append({
                    'feature': feature,
                    'mean': data[feature].mean(),
                    'std': data[feature].std(),
                    'min': data[feature].min(),
                    'max': data[feature].max(),
                    'missing%': data[feature].isnull().sum() / len(data) * 100
                })
        
        if stats:
            self.logger.info("Key feature statistics:")
            for stat in stats:
                self.logger.info(f"  {stat['feature']}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, missing={stat['missing%']:.2f}%")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Pipeline state dictionary
        """
        return self.pipeline_state.copy()
    
    def stop_pipeline(self):
        """Stop the pipeline execution."""
        self.logger.warning("Pipeline stop requested")
        self.pipeline_state['status'] = 'stopped'
        self.pipeline_state['end_time'] = datetime.now()