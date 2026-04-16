
import time
import logging
import functools
from typing import Callable, Any, Tuple, Type, Optional

logger = logging.getLogger(__name__)

def retry(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    logger_name: Optional[str] = None
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        exceptions: Tuple of exceptions to catch and retry.
        tries: Maximum number of attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay after each attempt.
        max_delay: Maximum delay in seconds.
        logger_name: Name of the logger to use.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logging.getLogger(logger_name) if logger_name else logger
            
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    msg = f"Retrying {func.__name__} in {_delay}s (Tries left: {_tries-1}): {e}"
                    log.warning(msg)
                    time.sleep(_delay)
                    _tries -= 1
                    _delay = min(_delay * backoff, max_delay)
            
            return func(*args, **kwargs)  # Last attempt without catching
        return wrapper
    return decorator
