import time
import functools

from loguru import logger


def retry(max_retries=3, backoff_factor=1):
    """
    Decorator to retry a function using exponential backoff.

    Args:
        max_retries (int): Maximum number of retries.
        backoff_factor (float): Factor by which the delay increases (in seconds).

    Returns:
        Decorated function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    delay = backoff_factor * (2 ** (retries - 1))
                    logger.exception(f"Retry {retries}/{max_retries} after {delay:.2f}s")
                    time.sleep(delay)

        return wrapper

    return decorator
