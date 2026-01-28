
import logging
import time
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger("Trace")

class Tracer:
    """
    Simple tracing wrapper.
    In future, this can be hooked to OpenTelemetry/Jaeger.
    """

    @staticmethod
    @contextmanager
    def span(name: str, attributes: dict = None):
        """
        Context manager for tracing a span.
        """
        start_time = time.time()
        # logger.debug(f"[TRACE-START] {name} {attributes or ''}")
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(f"[TRACE] {name} completed in {duration:.4f}s")
            # Here we would emit to Jaeger

def trace_span(name: str):
    """Decorator for tracing functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Tracer.span(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
