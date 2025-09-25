"""
Base Collector Abstract Class
Defines the interface for all data collectors in the pipeline
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import pandas as pd
from datetime import datetime, timedelta
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import json
import pickle
from pathlib import Path


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.
    Implements common functionality like caching, retry logic, and logging.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        cache_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize base collector.
        
        Args:
            config: Configuration dictionary
            cache_dir: Directory for caching data
            logger: Logger instance
        """
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Performance metrics
        self.metrics = {
            'requests_made': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0
        }
        
        # Initialize collector-specific configuration
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize collector-specific settings."""
        pass
    
    @abstractmethod
    def collect(
        self, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        **kwargs
    ) -> pd.DataFrame:
        """
        Main collection method that must be implemented by subclasses.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            **kwargs: Additional collector-specific parameters
            
        Returns:
            DataFrame with collected data
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate collected data for basic sanity checks.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data passes validation, False otherwise
        """
        pass
    
    def get_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            params: Parameters dictionary
            
        Returns:
            Hash string for cache key
        """
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()
    
    def load_from_cache(
        self, 
        cache_key: str, 
        ttl_hours: Optional[float] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load data from cache if exists and is valid.
        
        Args:
            cache_key: Cache key
            ttl_hours: Time to live in hours
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_file = self.cache_dir / f"{self.__class__.__name__}_{cache_key}.pkl"
        
        if not cache_file.exists():
            self.metrics['cache_misses'] += 1
            return None
        
        # Check TTL if specified
        if ttl_hours:
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age > timedelta(hours=ttl_hours):
                self.logger.debug(f"Cache expired for key {cache_key}")
                self.metrics['cache_misses'] += 1
                return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            self.logger.debug(f"Cache hit for key {cache_key}")
            self.metrics['cache_hits'] += 1
            return data
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            self.metrics['cache_misses'] += 1
            return None
    
    def save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """
        Save data to cache.
        
        Args:
            cache_key: Cache key
            data: DataFrame to cache
        """
        cache_file = self.cache_dir / f"{self.__class__.__name__}_{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug(f"Data cached with key {cache_key}")
        except Exception as e:
            self.logger.error(f"Error saving to cache: {e}")
    
    def clean_cache(self, days_old: int = 7) -> None:
        """
        Clean old cache files.
        
        Args:
            days_old: Remove files older than this many days
        """
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        for cache_file in self.cache_dir.glob(f"{self.__class__.__name__}_*.pkl"):
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_time < cutoff_time:
                cache_file.unlink()
                self.logger.info(f"Removed old cache file: {cache_file.name}")
    
    def standardize_dataframe(
        self, 
        df: pd.DataFrame,
        date_column: str = None,
        timezone: str = 'America/Sao_Paulo'
    ) -> pd.DataFrame:
        """
        Standardize DataFrame format.
        
        Args:
            df: DataFrame to standardize
            date_column: Name of date column
            timezone: Timezone to localize to
            
        Returns:
            Standardized DataFrame
        """
        df = df.copy()
        
        # Set date index if specified
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Localize to specified timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone)
        else:
            df.index = df.index.tz_convert(timezone)
        
        # Sort by index
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def handle_missing_data(
        self, 
        df: pd.DataFrame,
        method: str = 'forward_fill',
        limit: int = 5
    ) -> pd.DataFrame:
        """
        Handle missing data in DataFrame.
        
        Args:
            df: DataFrame with potential missing data
            method: Method to handle missing data
            limit: Maximum number of consecutive NaNs to fill
            
        Returns:
            DataFrame with missing data handled
        """
        df = df.copy()
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill', limit=limit)
        elif method == 'backward_fill':
            df = df.fillna(method='bfill', limit=limit)
        elif method == 'interpolate':
            df = df.interpolate(method='linear', limit=limit)
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Log missing data statistics
        missing_count = df.isnull().sum()
        if missing_count.any():
            self.logger.warning(f"Missing data after handling: {missing_count[missing_count > 0].to_dict()}")
        
        return df
    
    def merge_timeframes(
        self,
        high_freq: pd.DataFrame,
        low_freq: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Merge data from different timeframes.
        
        Args:
            high_freq: Higher frequency data (e.g., daily)
            low_freq: Lower frequency data (e.g., monthly)
            method: Method to align frequencies
            
        Returns:
            Merged DataFrame at higher frequency
        """
        # Ensure both have datetime index
        high_freq = self.standardize_dataframe(high_freq)
        low_freq = self.standardize_dataframe(low_freq)
        
        # Resample low frequency to high frequency
        if method == 'forward_fill':
            low_freq_resampled = low_freq.resample('D').ffill()
        elif method == 'backward_fill':
            low_freq_resampled = low_freq.resample('D').bfill()
        elif method == 'interpolate':
            low_freq_resampled = low_freq.resample('D').interpolate()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Merge on index
        merged = pd.concat([high_freq, low_freq_resampled], axis=1)
        
        return merged
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def make_request(
        self, 
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: int = 30
    ) -> Any:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: URL to request
            params: Query parameters
            headers: Request headers
            timeout: Request timeout in seconds
            
        Returns:
            Response object
        """
        import requests
        
        self.metrics['requests_made'] += 1
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            self.metrics['requests_failed'] += 1
            self.logger.error(f"Request failed: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get collector metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset collector metrics."""
        for key in self.metrics:
            self.metrics[key] = 0
    
    def __repr__(self) -> str:
        """String representation of collector."""
        return f"{self.__class__.__name__}(cache_dir={self.cache_dir})"