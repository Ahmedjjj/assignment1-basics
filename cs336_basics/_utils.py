import json
import logging
import os
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

_logger = logging.getLogger(__name__)


def encode(text: str) -> bytes:
    return text.encode("utf-8")


F = TypeVar("F", bound=Callable)


def log_time(
    step_name: str | None,
    logger: logging.Logger | None = None,
) -> Callable[[F], F]:
    logger = logger or _logger

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = step_name or func.__name__
            start = time.perf_counter()
            logger.info("%s started", name)
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.info("%s took %.8f seconds", name, elapsed)

        return wrapper  # type: ignore

    return decorator


def save_as_json(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_from_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def configure_logging(logger: logging.Logger | None = None, file: str | os.PathLike | None = None) -> None:
    logger = logger or logging.getLogger("cs336_basics")

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if file is not None:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
