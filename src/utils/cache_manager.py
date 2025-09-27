"""
Cache Manager Module
Handles caching for the USD/BRL pipeline with Redis and local file fallback
"""

import pickle
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Union
import logging
import redis
import pandas as pd
import numpy as np
from functools import wraps
import time


class CacheManager:
    """
    Manages caching with Redis primary and file system fallback.
    
    Features:
    - Redis for distributed caching
    - Local file system fallback
    - TTL support
    - Compression for large objects
    - Cache statistics
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize cache manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Cache configuration
        cache_config = config.get('database', {}).get('cache', {})
        
        # Redis configuration
        self.redis_enabled = cache_config.get('type') == 'redis'
        self.redis_client = None
        
        if self.redis_enabled:
            try:
                self.redis_client = redis.Redis(
                    host=cache_config.get('host', 'localhost'),
                    port=cache_config.get('port', 6379),
                    db=cache_config.get('db', 0),
                    password=cache_config.get('password'),
                    decode_responses=False  # We'll handle encoding
                )
                # Test connection
                self.redis_client.ping()
                self.logger.info("Redis cache connected")
            except Exception as e:
                self.logger.warning(f"Redis connection failed, using file cache: {e}")
                self.redis_enabled = False
                self.redis_client = None
        
        # File cache configuration
        self.cache_dir = Path(config.get('output', {}).get('base_path', 'data')) / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.default_ttl = config.get('collection', {}).get('cache_ttl_hours', 1) * 3600
        self.compression_threshold = 1024 * 1024  # 1MB
        self.max_cache_size = config.get('performance', {}).get('cache_size_mb', 500) * 1024 * 1024
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0,
            'redis_hits': 0,
            'file_hits': 0
        }
    
    def get(
        self,
        key: str,
        default: Any = None,
        ttl_override: Optional[int] = None
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            ttl_override: Override TTL check (seconds)
            
        Returns:
            Cached value or default
        """
        try:
            # Try Redis first
            if self.redis_enabled and self.redis_client:
                value = self._get_redis(key)
                if value is not None:
                    self.stats['hits'] += 1
                    self.stats['redis_hits'] += 1
                    self.logger.debug(f"Cache hit (Redis): {key}")
                    return value
            
            # Fallback to file cache
            value = self._get_file(key, ttl_override)
            if value is not None:
                self.stats['hits'] += 1
                self.stats['file_hits'] += 1
                self.logger.debug(f"Cache hit (File): {key}")
                return value
            
            # Cache miss
            self.stats['misses'] += 1
            self.logger.debug(f"Cache miss: {key}")
            return default
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Cache get error for {key}: {e}")
            return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            ttl = ttl or self.default_ttl
            
            # Try Redis first
            if self.redis_enabled and self.redis_client:
                success = self._set_redis(key, value, ttl)
                if success:
                    self.stats['sets'] += 1
                    self.logger.debug(f"Cache set (Redis): {key}")
            
            # Always set in file cache as backup
            success = self._set_file(key, value, ttl)
            if success:
                self.stats['sets'] += 1
                self.logger.debug(f"Cache set (File): {key}")
            
            return success
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Cache set error for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        try:
            # Delete from Redis
            if self.redis_enabled and self.redis_client:
                self.redis_client.delete(key)
            
            # Delete from file cache
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                cache_file.unlink()
            
            self.stats['deletes'] += 1
            self.logger.debug(f"Cache deleted: {key}")
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Cache delete error for {key}: {e}")
            return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            pattern: Optional pattern to match keys
            
        Returns:
            Number of entries cleared
        """
        count = 0
        
        try:
            # Clear from Redis
            if self.redis_enabled and self.redis_client:
                if pattern:
                    keys = self.redis_client.keys(f"{pattern}*")
                    if keys:
                        count += self.redis_client.delete(*keys)
                else:
                    self.redis_client.flushdb()
            
            # Clear from file cache
            if pattern:
                for cache_file in self.cache_dir.glob(f"{pattern}*.pkl"):
                    cache_file.unlink()
                    count += 1
            else:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    count += 1
            
            self.logger.info(f"Cleared {count} cache entries")
            return count
            
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        
        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        if total_requests > 0:
            stats['hit_rate'] = stats['hits'] / total_requests * 100
        else:
            stats['hit_rate'] = 0
        
        # Get cache size
        stats['file_cache_size_mb'] = self._get_file_cache_size() / 1024 / 1024
        
        if self.redis_enabled and self.redis_client:
            try:
                info = self.redis_client.info('memory')
                stats['redis_memory_mb'] = info.get('used_memory', 0) / 1024 / 1024
            except:
                pass
        
        return stats
    
    def _get_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            self.logger.debug(f"Redis get error: {e}")
            return None
    
    def _set_redis(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            # Serialize value
            data = pickle.dumps(value)
            
            # Set with TTL
            self.redis_client.setex(key, ttl, data)
            return True
            
        except Exception as e:
            self.logger.debug(f"Redis set error: {e}")
            return False
    
    def _get_file(self, key: str, ttl_override: Optional[int] = None) -> Optional[Any]:
        """Get value from file cache."""
        cache_file = self._get_cache_file_path(key)
        
        if not cache_file.exists():
            return None
        
        try:
            # Check TTL
            if ttl_override is not None:
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age > ttl_override:
                    self.logger.debug(f"File cache expired: {key}")
                    return None
            
            # Load data
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check embedded TTL
            if isinstance(data, dict) and '_cache_metadata' in data:
                metadata = data['_cache_metadata']
                if datetime.now() > metadata['expires']:
                    self.logger.debug(f"File cache expired (embedded): {key}")
                    return None
                return data['value']
            
            return data
            
        except Exception as e:
            self.logger.debug(f"File cache read error: {e}")
            return None
    
    def _set_file(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in file cache."""
        cache_file = self._get_cache_file_path(key)
        
        try:
            # Add metadata
            data = {
                '_cache_metadata': {
                    'key': key,
                    'created': datetime.now(),
                    'expires': datetime.now() + timedelta(seconds=ttl),
                    'ttl': ttl
                },
                'value': value
            }
            
            # Save to file
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            self.logger.debug(f"File cache write error: {e}")
            return False
    
    def _get_cache_file_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Hash the key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def _get_file_cache_size(self) -> int:
        """Get total size of file cache in bytes."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            total_size += cache_file.stat().st_size
        return total_size
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned
        """
        count = 0
        
        # Clean file cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict) and '_cache_metadata' in data:
                    if datetime.now() > data['_cache_metadata']['expires']:
                        cache_file.unlink()
                        count += 1
            except:
                # Remove corrupted files
                cache_file.unlink()
                count += 1
        
        self.logger.info(f"Cleaned up {count} expired cache entries")
        return count


def cache_result(
    ttl: int = 3600,
    key_prefix: str = None,
    ignore_args: list = None
):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        ignore_args: List of argument names to ignore in cache key
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create cache manager
            cache_manager = getattr(wrapper, '_cache_manager', None)
            if cache_manager is None:
                # Create a simple cache manager
                cache_manager = LocalCache()
                wrapper._cache_manager = cache_manager
            
            # Generate cache key
            cache_key = _generate_cache_key(
                func,
                args,
                kwargs,
                key_prefix,
                ignore_args
            )
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class LocalCache:
    """Simple local memory cache for decorator use."""
    
    def __init__(self):
        self.cache = {}
        self.expiry = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            if datetime.now() < self.expiry[key]:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.expiry[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int):
        """Set value in cache."""
        self.cache[key] = value
        self.expiry[key] = datetime.now() + timedelta(seconds=ttl)


def _generate_cache_key(
    func,
    args,
    kwargs,
    key_prefix: Optional[str] = None,
    ignore_args: Optional[list] = None
) -> str:
    """Generate cache key for function call."""
    ignore_args = ignore_args or []
    
    # Start with function name
    key_parts = [key_prefix or func.__name__]
    
    # Add args
    for i, arg in enumerate(args):
        if i not in ignore_args:
            key_parts.append(str(arg))
    
    # Add kwargs
    for k, v in sorted(kwargs.items()):
        if k not in ignore_args:
            key_parts.append(f"{k}={v}")
    
    # Create key
    key = ":".join(key_parts)
    
    # Hash if too long
    if len(key) > 200:
        key = hashlib.md5(key.encode()).hexdigest()
    
    return key