import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    bpe_count_max_workers: int = os.cpu_count() or 1
    results_folder: Path = Path("./results")
    overwrite_results: bool = False
    track_memory: bool = False


settings = Settings()
