"""
Preprocessing and data loading utilities for Yoochoose Session-Based Recommendation.

This module provides reusable primitives for reading raw clickstream data,
parsing ISO timestamps, deterministic row indexing, and detecting duplicate/repeated clicks.
"""

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard column names for Yoochoose clickstream dataset (no header in raw file)
RAW_COLUMN_NAMES = ["session_id", "timestamp", "item_id", "category"]

# Memory-efficient dtypes for raw reading
RAW_DTYPES = {
    "session_id": np.int64,
    "timestamp": str,
    "item_id": np.int64,
    "category": str,
}


def get_repo_root() -> Path:
    """
    Determine and return the absolute path to the repository root.
    Assumes this file is located at <repo_root>/src/utils/preprocessing.py.
    """
    return Path(__file__).resolve().parents[2]


def parse_iso_timestamps(
    timestamp_series: pd.Series,
) -> Tuple[pd.Series, pd.Series, int]:
    """
    Parse ISO 8601 UTC timestamp strings into datetime and epoch milliseconds.

    Args:
        timestamp_series: Pandas Series containing ISO timestamp strings (e.g. '2014-04-07T10:51:09.277Z').

    Returns:
        A tuple of (parsed_datetime_series, timestamp_ms_series, num_failures).
        - parsed_datetime_series: pd.Series of datetime64[ns, UTC]
        - timestamp_ms_series: pd.Series of int64 epoch milliseconds (or -1 if failed)
        - num_failures: count of timestamps that failed to parse
    """
    parsed = pd.to_datetime(
        timestamp_series,
        format="ISO8601",
        errors="coerce",
        utc=True,
    )
    num_failures = int(parsed.isna().sum())

    # Convert valid datetimes to epoch milliseconds int64
    valid_mask = parsed.notna()
    ts_ms_values = np.full(len(timestamp_series), -1, dtype=np.int64)
    if valid_mask.any():
        ts_ms_values[valid_mask.to_numpy()] = (
            parsed[valid_mask].astype("int64").to_numpy() // 1_000_000
        )

    ts_ms = pd.Series(ts_ms_values, index=timestamp_series.index, dtype=np.int64)
    return parsed, ts_ms, num_failures


def read_raw_clicks_chunked(
    file_path: Union[str, Path],
    chunksize: int = 2_000_000,
    add_row_index: bool = True,
) -> Iterator[pd.DataFrame]:
    """
    Read raw Yoochoose clicks file in chunks with standardized schema and parsed timestamps.

    Args:
        file_path: Path to the raw yoochoose_clicks.dat file.
        chunksize: Number of rows per chunk. Default is 2,000,000.
        add_row_index: Whether to add a global 'original_row_number' column for tie-breaking.

    Yields:
        pd.DataFrame for each chunk containing:
            - session_id (int64)
            - timestamp (str, original raw ISO string)
            - item_id (int64)
            - category (str)
            - parsed_datetime (datetime64[ns, UTC])
            - timestamp_ms (int64)
            - parsing_failed (bool)
            - original_row_number (int64, if add_row_index=True)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")

    current_row_offset = 0
    for chunk in pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMN_NAMES,
        dtype=RAW_DTYPES,
        chunksize=chunksize,
        low_memory=False,
    ):
        num_rows = len(chunk)
        if add_row_index:
            chunk["original_row_number"] = np.arange(
                current_row_offset,
                current_row_offset + num_rows,
                dtype=np.int64,
            )

        parsed_dt, ts_ms, failures = parse_iso_timestamps(chunk["timestamp"])
        chunk["parsed_datetime"] = parsed_dt
        chunk["timestamp_ms"] = ts_ms
        chunk["parsing_failed"] = (ts_ms == -1)

        if failures > 0:
            logger.warning(
                "Found %d timestamp parsing failure(s) in chunk starting at row %d.",
                failures,
                current_row_offset,
            )

        current_row_offset += num_rows
        yield chunk


def detect_exact_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> pd.Series:
    """
    Identify exact duplicate rows within a DataFrame.

    Args:
        df: Input DataFrame.
        subset: Optional list of column names to check for duplication.
                Defaults to ['session_id', 'timestamp_ms', 'item_id', 'category'].

    Returns:
        Boolean Series indicating duplicate rows (True for duplicates after first occurrence).
    """
    if subset is None:
        subset = [
            col
            for col in ["session_id", "timestamp_ms", "item_id", "category"]
            if col in df.columns
        ]
    return df.duplicated(subset=subset, keep="first")


def detect_consecutive_repeats(
    df: pd.DataFrame,
    session_col: str = "session_id",
    item_col: str = "item_id",
) -> pd.Series:
    """
    Identify consecutive repeated items within the same session.

    Assumes the DataFrame is ordered chronologically / by tie-breaker within each session.
    Example:
        Item sequence: [A, A, B, A] -> [False, True, False, False]

    Args:
        df: Input DataFrame.
        session_col: Column name for session ID.
        item_col: Column name for item ID.

    Returns:
        Boolean Series where True indicates that the row is a consecutive repetition
        of the preceding item in the same session.
    """
    same_session = df[session_col] == df[session_col].shift(1)
    same_item = df[item_col] == df[item_col].shift(1)
    return same_session & same_item
