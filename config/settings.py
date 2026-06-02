"""Central configuration. Loads from environment variables."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    database_url: str
    nse_symbols: tuple = ("NIFTY", "BANKNIFTY")
    fetch_timeout_sec: int = 15
    ingestion_interval_sec: int = 300  # 5 min
    market_open: str = "09:15"
    market_close: str = "15:30"
    timezone: str = "Asia/Kolkata"
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )


def load() -> Config:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL env var required")
    return Config(database_url=db_url)
