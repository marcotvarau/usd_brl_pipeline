"""
Circuit Breaker Pattern Implementation
Provides fault tolerance for external API calls
"""

import time
import logging
from typing import Any, Callable, Optional, Dict
from enum import Enum
from functools import wraps


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying again (seconds)
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: When circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - not executing function")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful function call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        if self.state != CircuitState.CLOSED:
            self.logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self) -> None:
        """Handle failed function call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )
    
    def can_execute(self, operation_name: str = None) -> bool:
        """
        Check if the circuit breaker allows execution.
        
        Args:
            operation_name: Name of the operation (for logging)
            
        Returns:
            True if execution is allowed, False otherwise
        """
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.HALF_OPEN:
            return True  # Allow one attempt in half-open state
        else:  # OPEN state
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info(f"Circuit breaker moving to HALF_OPEN state for {operation_name}")
                return True
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout
        }
