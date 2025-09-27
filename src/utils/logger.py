"""
Logger Utility Module
Configures structured logging for the USD/BRL pipeline
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pythonjsonlogger import jsonlogger
import coloredlogs


class ContextFilter(logging.Filter):
    """Add context information to log records."""
    
    def __init__(self, context: Dict[str, Any] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        # Add context to record
        for key, value in self.context.items():
            setattr(record, key, value)
        
        # Add default fields
        record.timestamp = datetime.utcnow().isoformat()
        record.hostname = self._get_hostname()
        
        return True
    
    @staticmethod
    def _get_hostname():
        import socket
        return socket.gethostname()


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add custom fields
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Add location information
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add timestamp
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


def setup_logger(
    name: str = 'usd_brl_pipeline',
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_format: str = 'json',
    console_output: bool = True,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Setup a configured logger.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Format ('json' or 'text')
        console_output: Whether to output to console
        context: Additional context to add to logs
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Add context filter
    if context:
        logger.addFilter(ContextFilter(context))
    
    # Create formatters
    if log_format == 'json':
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            json_ensure_ascii=False
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if log_format == 'text' and sys.stdout.isatty():
            # Use colored output for terminal
            coloredlogs.install(
                level=log_level,
                logger=logger,
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_multi_logger(config: Dict[str, Any]) -> Dict[str, logging.Logger]:
    """
    Setup multiple loggers based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of loggers
    """
    loggers = {}
    
    # Main pipeline logger
    loggers['pipeline'] = setup_logger(
        name='pipeline',
        log_level=config.get('logging', {}).get('level', 'INFO'),
        log_file='logs/pipeline.log',
        log_format=config.get('logging', {}).get('format', 'json')
    )
    
    # Error logger
    loggers['errors'] = setup_logger(
        name='errors',
        log_level='ERROR',
        log_file='logs/errors.log',
        log_format='json'
    )
    
    # Validation logger
    loggers['validation'] = setup_logger(
        name='validation',
        log_level='INFO',
        log_file='logs/validation.log',
        log_format='json'
    )
    
    # Performance logger
    loggers['performance'] = setup_logger(
        name='performance',
        log_level='DEBUG',
        log_file='logs/performance.log',
        log_format='json'
    )
    
    return loggers


class LoggingContext:
    """Context manager for adding temporary logging context."""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize logging context.
        
        Args:
            logger: Logger instance
            **kwargs: Context key-value pairs
        """
        self.logger = logger
        self.context = kwargs
        self.old_filters = []
    
    def __enter__(self):
        """Enter context."""
        # Add context filter
        filter = ContextFilter(self.context)
        self.logger.addFilter(filter)
        self.old_filters.append(filter)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # Remove added filters
        for filter in self.old_filters:
            self.logger.removeFilter(filter)


class PerformanceLogger:
    """Logger for tracking performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, operation: str):
        """
        Start timing an operation.
        
        Args:
            operation: Operation name
        """
        self.timers[operation] = datetime.now()
        self.logger.debug(f"Started operation: {operation}")
    
    def end_timer(self, operation: str, metadata: Optional[Dict] = None):
        """
        End timing an operation and log the duration.
        
        Args:
            operation: Operation name
            metadata: Additional metadata to log
        """
        if operation not in self.timers:
            self.logger.warning(f"Timer not started for operation: {operation}")
            return
        
        start_time = self.timers[operation]
        duration = (datetime.now() - start_time).total_seconds()
        
        log_data = {
            'operation': operation,
            'duration_seconds': duration,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }
        
        if metadata:
            log_data.update(metadata)
        
        self.logger.info(f"Completed operation: {operation}", extra=log_data)
        
        del self.timers[operation]
    
    def log_metric(self, metric_name: str, value: float, unit: str = None, metadata: Optional[Dict] = None):
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            metadata: Additional metadata
        """
        log_data = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            log_data.update(metadata)
        
        self.logger.info(f"Metric: {metric_name}={value}{unit or ''}", extra=log_data)


class DataLogger:
    """Logger for data quality and statistics."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize data logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
    
    def log_dataframe_stats(self, df, name: str = "dataframe"):
        """
        Log statistics about a DataFrame.
        
        Args:
            df: pandas DataFrame
            name: Name for the DataFrame
        """
        import pandas as pd
        
        stats = {
            'name': name,
            'shape': list(df.shape),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'columns': list(df.columns)
        }
        
        if isinstance(df.index, pd.DatetimeIndex):
            stats['date_range'] = {
                'start': str(df.index.min()),
                'end': str(df.index.max()),
                'days': (df.index.max() - df.index.min()).days
            }
        
        self.logger.info(f"DataFrame stats for {name}", extra=stats)
    
    def log_data_quality(self, issues: list, passed: bool, name: str = "validation"):
        """
        Log data quality validation results.
        
        Args:
            issues: List of issues found
            passed: Whether validation passed
            name: Name of validation
        """
        quality_data = {
            'validation_name': name,
            'passed': passed,
            'issues_count': len(issues),
            'issues': issues[:10],  # Log first 10 issues
            'timestamp': datetime.now().isoformat()
        }
        
        if passed:
            self.logger.info(f"Data quality check passed: {name}", extra=quality_data)
        else:
            self.logger.warning(f"Data quality issues found: {name}", extra=quality_data)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to module name)
        
    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


# Example usage configuration
if __name__ == "__main__":
    # Setup example logger
    logger = setup_logger(
        name='example',
        log_level='DEBUG',
        log_file='logs/example.log',
        log_format='json',
        context={'environment': 'development', 'version': '1.0.0'}
    )
    
    # Example logging
    logger.info("Pipeline started", extra={'step': 'initialization'})
    logger.debug("Loading configuration")
    logger.warning("Missing optional parameter", extra={'parameter': 'cache_ttl'})
    
    # Performance logging example
    perf_logger = PerformanceLogger(logger)
    perf_logger.start_timer('data_collection')
    # ... do work ...
    perf_logger.end_timer('data_collection', metadata={'rows_collected': 1000})
    
    # Data logging example
    data_logger = DataLogger(logger)
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    data_logger.log_dataframe_stats(df, 'example_data')
    
    # Context example
    with LoggingContext(logger, operation='validation', user='admin'):
        logger.info("Running validation")
        # Logs will include operation and user context