import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, TypeVar

from cs336_basics.pretokenization_example import find_chunk_boundaries

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
        json.dump(obj, f, indent=2)


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


def read_in_chunks(path: str, split_token: str, num_chunks: int) -> Iterable[str]:
    with open(path, mode="rb") as buf:
        boundaries = find_chunk_boundaries(
            file=buf,
            desired_num_chunks=num_chunks,
            split_special_token=split_token.encode("utf-8"),
        )

        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            buf.seek(start)
            yield buf.read(end - start).decode("utf-8", errors="ignore")
