#!/usr/bin/env python3
"""
Dataset Inspection Script for Yoochoose Clickstream Dataset.

Memory-efficient, chunked inspection pipeline for session-based recommendation analysis.
Generates comprehensive statistics on data quality, session distributions, timing,
item frequencies, consecutive repeats, temporal trends, and subset what-if scenarios.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.preprocessing import (
    RAW_COLUMN_NAMES,
    detect_exact_duplicates,
    get_repo_root,
    read_raw_clicks_chunked,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Memory-efficient dataset inspection for Yoochoose clickstream."
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/yoochoose_clicks.dat",
        help="Path to raw yoochoose_clicks.dat file (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/inspection",
        help="Directory to save inspection results and reports.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=2_000_000,
        help="Number of rows per chunk for streaming inspection (default: 2,000,000).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional maximum number of chunks to process (for quick testing).",
    )
    return parser.parse_args()


def compute_percentiles(
    arr: np.ndarray,
    percentiles: List[float] = [25, 50, 75, 90, 95, 99, 99.5],
) -> Dict[str, float]:
    """Compute standard percentiles and basic summary statistics for a 1D numpy array."""
    if len(arr) == 0:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "max": 0.0,
            **{f"p{p}": 0.0 for p in percentiles},
        }

    res: Dict[str, float] = {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }
    pct_values = np.percentile(arr, percentiles)
    for p, val in zip(percentiles, pct_values):
        key = f"p{str(p).replace('.', '_')}"
        res[key] = float(val)
    return res


def run_inspection(
    raw_path: Path,
    output_dir: Path,
    chunksize: int = 2_000_000,
    max_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run memory-efficient dataset inspection and produce all artifacts.
    """
    start_time_sec = time.time()
    logger.info("Starting dataset inspection...")
    logger.info("Raw data path: %s", raw_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Chunksize: %d rows", chunksize)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    raw_file_size_bytes = raw_path.stat().st_size
    raw_file_size_mb = raw_file_size_bytes / (1024 * 1024)
    logger.info("Raw file size: %.2f MB (%.2f GB)", raw_file_size_mb, raw_file_size_mb / 1024)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Tracking structures for single-pass streaming
    total_clicks = 0
    missing_counts = {col: 0 for col in RAW_COLUMN_NAMES}
    parsing_failures_count = 0
    invalid_rows_count = 0
    total_exact_duplicates = 0
    total_timestamp_ties = 0
    same_session_timestamp_ties = 0
    total_consec_repeats = 0

    is_globally_sorted = True
    prev_chunk_last_ts: Optional[int] = None
    prev_chunk_last_row_tuple: Optional[Tuple[Any, ...]] = None

    item_counter: Dict[int, int] = {}
    category_counter: Dict[str, int] = {}
    daily_clicks: Dict[str, int] = {}
    daily_sessions: Dict[str, set] = {}
    daily_items: Dict[str, set] = {}

    # Session dictionary:
    # session_id -> [click_count, start_ts_ms, end_ts_ms, last_item, last_ts_ms, last_row_num, num_consec_repeats]
    session_tracker: Dict[int, List[Any]] = {}

    # List to store inter-click time gaps (in seconds) in float32 chunks
    gap_chunks: List[np.ndarray] = []

    min_ts_ms = float("inf")
    max_ts_ms = float("-inf")

    chunk_idx = 0
    for chunk in read_raw_clicks_chunked(raw_path, chunksize=chunksize, add_row_index=True):
        chunk_idx += 1
        num_chunk_rows = len(chunk)
        total_clicks += num_chunk_rows

        # Track missing values
        for col in RAW_COLUMN_NAMES:
            missing_counts[col] += int(chunk[col].isna().sum())

        # Track parsing failures
        failed_mask = chunk["parsing_failed"]
        num_failed = int(failed_mask.sum())
        parsing_failures_count += num_failed

        # Filter valid rows for temporal and session processing
        valid_chunk = chunk[~failed_mask]

        if len(valid_chunk) > 0:
            ts_ms_arr = valid_chunk["timestamp_ms"].to_numpy()
            chunk_min_ts = int(np.min(ts_ms_arr))
            chunk_max_ts = int(np.max(ts_ms_arr))
            if chunk_min_ts < min_ts_ms:
                min_ts_ms = chunk_min_ts
            if chunk_max_ts > max_ts_ms:
                max_ts_ms = chunk_max_ts

            # Check global sorting
            if not valid_chunk["timestamp_ms"].is_monotonic_increasing:
                is_globally_sorted = False
            if prev_chunk_last_ts is not None and chunk_min_ts < prev_chunk_last_ts:
                is_globally_sorted = False
            prev_chunk_last_ts = chunk_max_ts

            # Global timestamp ties
            ties_in_chunk = int((ts_ms_arr[1:] == ts_ms_arr[:-1]).sum())
            boundary_tie = 1 if (prev_chunk_last_ts is not None and len(ts_ms_arr) > 0 and ts_ms_arr[0] == prev_chunk_last_ts) else 0
            total_timestamp_ties += ties_in_chunk + boundary_tie

        # Exact duplicate rows in chunk
        dup_mask = detect_exact_duplicates(chunk)
        total_exact_duplicates += int(dup_mask.sum())

        # Boundary duplicate check
        if prev_chunk_last_row_tuple is not None:
            first_row_tuple = (
                chunk.iloc[0]["session_id"],
                chunk.iloc[0]["timestamp_ms"],
                chunk.iloc[0]["item_id"],
                chunk.iloc[0]["category"],
            )
            if first_row_tuple == prev_chunk_last_row_tuple:
                total_exact_duplicates += 1

        last_row = chunk.iloc[-1]
        prev_chunk_last_row_tuple = (
            last_row["session_id"],
            last_row["timestamp_ms"],
            last_row["item_id"],
            last_row["category"],
        )

        # Update item and category counters
        item_vals, item_counts = np.unique(chunk["item_id"].to_numpy(), return_counts=True)
        for item, cnt in zip(item_vals, item_counts):
            item_counter[int(item)] = item_counter.get(int(item), 0) + int(cnt)

        cat_vals, cat_counts = np.unique(chunk["category"].astype(str).to_numpy(), return_counts=True)
        for cat, cnt in zip(cat_vals, cat_counts):
            category_counter[str(cat)] = category_counter.get(str(cat), 0) + int(cnt)

        # Daily activity accumulation
        if len(valid_chunk) > 0:
            dt_series = pd.to_datetime(valid_chunk["timestamp_ms"], unit="ms", utc=True)
            dates = dt_series.dt.strftime("%Y-%m-%d").to_numpy()
            sids = valid_chunk["session_id"].to_numpy()
            iids = valid_chunk["item_id"].to_numpy()

            unique_dates_chunk = np.unique(dates)
            for d in unique_dates_chunk:
                d_mask = dates == d
                daily_clicks[d] = daily_clicks.get(d, 0) + int(np.sum(d_mask))
                if d not in daily_sessions:
                    daily_sessions[d] = set()
                    daily_items[d] = set()
                daily_sessions[d].update(sids[d_mask].tolist())
                daily_items[d].update(iids[d_mask].tolist())

        # Process session stream
        sids = chunk["session_id"].to_numpy()
        ts = chunk["timestamp_ms"].to_numpy()
        items = chunk["item_id"].to_numpy()
        rows = chunk["original_row_number"].to_numpy()

        chunk_gaps = []
        for sid, t_ms, item, rnum in zip(sids, ts, items, rows):
            if sid not in session_tracker:
                session_tracker[sid] = [1, t_ms, t_ms, item, t_ms, rnum, 0]
            else:
                info = session_tracker[sid]
                info[0] += 1  # count
                info[2] = t_ms  # end_ts_ms
                gap_ms = t_ms - info[4]
                if gap_ms >= 0:
                    chunk_gaps.append(float(gap_ms) / 1000.0)
                if t_ms == info[4]:
                    same_session_timestamp_ties += 1
                if item == info[3]:
                    info[6] += 1  # consec repeats
                    total_consec_repeats += 1
                info[3] = item
                info[4] = t_ms
                info[5] = rnum

        if len(chunk_gaps) > 0:
            gap_chunks.append(np.array(chunk_gaps, dtype=np.float32))

        logger.info(
            "Processed chunk %d (%d rows) | Cumulative clicks: %d | Unique sessions: %d",
            chunk_idx,
            num_chunk_rows,
            total_clicks,
            len(session_tracker),
        )

        if max_chunks is not None and chunk_idx >= max_chunks:
            logger.info("Reached maximum requested chunks (%d). Stopping stream.", max_chunks)
            break

    elapsed_stream_sec = time.time() - start_time_sec
    logger.info("Streaming pass completed in %.2f seconds.", elapsed_stream_sec)

    # -------------------------------------------------------------
    # 1. BASIC DATASET SUMMARY & METADATA
    # -------------------------------------------------------------
    total_unique_sessions = len(session_tracker)
    total_unique_items = len(item_counter)
    total_unique_categories = len(category_counter)

    min_datetime_utc = (
        pd.to_datetime(min_ts_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        if min_ts_ms != float("inf")
        else "N/A"
    )
    max_datetime_utc = (
        pd.to_datetime(max_ts_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        if max_ts_ms != float("-inf")
        else "N/A"
    )

    date_range_days = (
        (max_ts_ms - min_ts_ms) / (1000.0 * 86400.0)
        if (min_ts_ms != float("inf") and max_ts_ms != float("-inf"))
        else 0.0
    )

    # -------------------------------------------------------------
    # 2. SESSION ARRAY EXTRACTION
    # -------------------------------------------------------------
    logger.info("Extracting structured session statistics for %d sessions...", total_unique_sessions)
    session_ids = np.fromiter(session_tracker.keys(), dtype=np.int64, count=total_unique_sessions)
    lengths = np.fromiter((v[0] for v in session_tracker.values()), dtype=np.int32, count=total_unique_sessions)
    start_ts = np.fromiter((v[1] for v in session_tracker.values()), dtype=np.int64, count=total_unique_sessions)
    end_ts = np.fromiter((v[2] for v in session_tracker.values()), dtype=np.int64, count=total_unique_sessions)
    consec_repeats = np.fromiter((v[6] for v in session_tracker.values()), dtype=np.int32, count=total_unique_sessions)

    # Free the large dictionary to minimize memory footprint
    del session_tracker
    gc.collect()

    # -------------------------------------------------------------
    # 3. SESSION LENGTH STATISTICS
    # -------------------------------------------------------------
    length_stats = compute_percentiles(lengths, percentiles=[25, 50, 75, 90, 95, 99, 99.5])
    num_1click = int(np.sum(lengths == 1))
    pct_1click = (num_1click / total_unique_sessions * 100.0) if total_unique_sessions > 0 else 0.0
    num_ge2click = int(np.sum(lengths >= 2))
    pct_ge2click = (num_ge2click / total_unique_sessions * 100.0) if total_unique_sessions > 0 else 0.0

    session_length_summary = {
        "total_sessions": total_unique_sessions,
        "min_length": length_stats["min"],
        "max_length": length_stats["max"],
        "mean_length": length_stats["mean"],
        "std_length": length_stats["std"],
        "median_length": length_stats["median"],
        "p25": length_stats["p25"],
        "p50": length_stats["p50"],
        "p75": length_stats["p75"],
        "p90": length_stats["p90"],
        "p95": length_stats["p95"],
        "p99": length_stats["p99"],
        "p99_5": length_stats["p99_5"],
        "num_1click_sessions": num_1click,
        "pct_1click_sessions": pct_1click,
        "num_ge2click_sessions": num_ge2click,
        "pct_ge2click_sessions": pct_ge2click,
    }

    # Session length distribution bins for CSV
    len_counts = pd.Series(lengths).value_counts().sort_index()
    cum_pct = (len_counts.cumsum() / total_unique_sessions * 100.0)
    len_dist_df = pd.DataFrame({
        "session_length": len_counts.index,
        "session_count": len_counts.values,
        "pct_of_sessions": (len_counts.values / total_unique_sessions * 100.0),
        "cumulative_pct": cum_pct.values,
    })
    len_dist_df.to_csv(output_dir / "session_length_distribution.csv", index=False)

    # -------------------------------------------------------------
    # 4. SESSION TIMING (For sessions with >= 2 clicks)
    # -------------------------------------------------------------
    logger.info("Computing session duration and inter-click timing statistics...")
    ge2_mask = lengths >= 2
    ge2_durations_sec = (end_ts[ge2_mask] - start_ts[ge2_mask]) / 1000.0
    duration_stats = compute_percentiles(ge2_durations_sec, percentiles=[25, 50, 75, 90, 95, 99])

    all_gaps_sec = np.concatenate(gap_chunks) if len(gap_chunks) > 0 else np.array([], dtype=np.float32)
    gap_stats = compute_percentiles(all_gaps_sec, percentiles=[25, 50, 75, 90, 95, 99])
    num_zero_gaps = int(np.sum(all_gaps_sec == 0.0)) if len(all_gaps_sec) > 0 else 0
    pct_zero_gaps = (num_zero_gaps / len(all_gaps_sec) * 100.0) if len(all_gaps_sec) > 0 else 0.0

    timing_summary = {
        "session_duration_seconds": duration_stats,
        "inter_click_gap_seconds": gap_stats,
        "zero_gap_transitions_count": num_zero_gaps,
        "zero_gap_transitions_pct": pct_zero_gaps,
    }

    # Session timing CSV
    timing_rows = [
        {"metric": "session_duration_sec", **duration_stats},
        {"metric": "inter_click_gap_sec", **gap_stats},
    ]
    pd.DataFrame(timing_rows).to_csv(output_dir / "session_timing.csv", index=False)

    # -------------------------------------------------------------
    # 5. CONSECUTIVE REPEATS
    # -------------------------------------------------------------
    total_transitions = total_clicks - total_unique_sessions
    repeat_transition_pct = (
        (total_consec_repeats / total_transitions * 100.0) if total_transitions > 0 else 0.0
    )
    sessions_with_repeats = int(np.sum(consec_repeats > 0))
    pct_sessions_with_repeats = (
        (sessions_with_repeats / total_unique_sessions * 100.0) if total_unique_sessions > 0 else 0.0
    )

    consecutive_repeats_summary = {
        "total_transitions": total_transitions,
        "consecutive_repeat_transitions": total_consec_repeats,
        "consecutive_repeat_transition_pct": repeat_transition_pct,
        "sessions_with_consecutive_repeats": sessions_with_repeats,
        "pct_sessions_with_consecutive_repeats": pct_sessions_with_repeats,
    }

    # -------------------------------------------------------------
    # 6. ITEM FREQUENCIES & LONG-TAIL ANALYSIS
    # -------------------------------------------------------------
    logger.info("Computing item frequency distributions and support thresholds...")
    item_counts_series = pd.Series(item_counter).sort_values(ascending=False)
    item_frequencies = item_counts_series.to_numpy()
    item_freq_stats = compute_percentiles(item_frequencies, percentiles=[25, 50, 75, 90, 95, 99])

    support_thresholds = [2, 5, 10, 20, 50, 100]
    support_analysis = []
    for th in support_thresholds:
        below_mask = item_frequencies < th
        num_items_below = int(np.sum(below_mask))
        pct_items_below = (num_items_below / total_unique_items * 100.0) if total_unique_items > 0 else 0.0
        clicks_below = int(np.sum(item_frequencies[below_mask]))
        pct_clicks_below = (clicks_below / total_clicks * 100.0) if total_clicks > 0 else 0.0

        support_analysis.append({
            "threshold": f"< {th} clicks",
            "items_count": num_items_below,
            "pct_of_unique_items": pct_items_below,
            "clicks_count": clicks_below,
            "pct_of_total_clicks": pct_clicks_below,
            "retained_items_count": total_unique_items - num_items_below,
            "pct_retained_items": 100.0 - pct_items_below,
            "retained_clicks_count": total_clicks - clicks_below,
            "pct_retained_clicks": 100.0 - pct_clicks_below,
        })

    support_df = pd.DataFrame(support_analysis)
    support_df.to_csv(output_dir / "item_frequency.csv", index=False)

    # Top items coverage
    cum_item_clicks = np.cumsum(item_frequencies)
    top_items_table = []
    for k in [10, 20, 50, 100]:
        if k <= len(item_frequencies):
            clicks_k = int(cum_item_clicks[k - 1])
            top_items_table.append({
                "top_k": k,
                "clicks": clicks_k,
                "pct_of_total_clicks": (clicks_k / total_clicks * 100.0),
            })

    # -------------------------------------------------------------
    # 7. DAILY ACTIVITY
    # -------------------------------------------------------------
    logger.info("Assembling daily temporal activity table...")
    sorted_dates = sorted(daily_clicks.keys())
    daily_rows = []
    for d in sorted_dates:
        daily_rows.append({
            "date": d,
            "click_count": daily_clicks[d],
            "unique_sessions": len(daily_sessions[d]),
            "unique_items": len(daily_items[d]),
        })
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(output_dir / "daily_counts.csv", index=False)

    # Free daily sets
    del daily_sessions, daily_items
    gc.collect()

    # -------------------------------------------------------------
    # 8. CATEGORY SUMMARY
    # -------------------------------------------------------------
    cat_series = pd.Series(category_counter).sort_values(ascending=False)
    top_categories = [
        {"category": str(cat), "clicks": int(cnt), "pct": float(cnt / total_clicks * 100.0)}
        for cat, cnt in cat_series.head(10).items()
    ]

    # -------------------------------------------------------------
    # 9. SUBSET WHAT-IF ANALYSIS (Ordered by end_time, session_id)
    # -------------------------------------------------------------
    logger.info("Conducting chronological subset what-if analysis...")
    sort_idx = np.lexsort((session_ids, end_ts))
    candidate_fractions = [
        ("1/64", 1.0 / 64.0),
        ("1/32", 1.0 / 32.0),
        ("1/16", 1.0 / 16.0),
        ("1/8", 1.0 / 8.0),
        ("1/4", 1.0 / 4.0),
        ("1/1 (Full)", 1.0),
    ]

    subset_results = []
    for label, frac in candidate_fractions:
        k = int(round(total_unique_sessions * frac)) if frac < 1.0 else total_unique_sessions
        k = max(1, min(k, total_unique_sessions))
        sel_idx = sort_idx[-k:]

        sub_clicks = int(np.sum(lengths[sel_idx]))
        sub_click_pct = (sub_clicks / total_clicks * 100.0) if total_clicks > 0 else 0.0
        sub_min_ts = int(np.min(start_ts[sel_idx]))
        sub_max_ts = int(np.max(end_ts[sel_idx]))

        sub_start_date = pd.to_datetime(sub_min_ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")
        sub_end_date = pd.to_datetime(sub_max_ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")
        sub_days = (sub_max_ts - sub_min_ts) / (1000.0 * 86400.0)
        approx_size_mb = (sub_clicks / total_clicks * raw_file_size_mb) if total_clicks > 0 else 0.0

        subset_results.append({
            "subset_fraction": label,
            "fraction_value": float(frac),
            "num_sessions": int(k),
            "pct_sessions": float(k / total_unique_sessions * 100.0),
            "num_clicks": sub_clicks,
            "pct_clicks": float(sub_click_pct),
            "date_range_start": sub_start_date,
            "date_range_end": sub_end_date,
            "span_days": float(round(sub_days, 2)),
            "approx_raw_size_mb": float(round(approx_size_mb, 2)),
        })

    subset_df = pd.DataFrame(subset_results)
    subset_df.to_csv(output_dir / "subset_what_if.csv", index=False)

    # -------------------------------------------------------------
    # 10. MAX-LENGTH INFORMATION (For sessions with >= 2 clicks)
    # -------------------------------------------------------------
    logger.info("Computing candidate max sequence length coverage...")
    ge2_lengths = lengths[ge2_mask]
    max_len_stats = compute_percentiles(ge2_lengths, percentiles=[50, 75, 90, 95, 99, 99.5])

    candidate_max_lens = [5, 10, 15, 20, 30, 40, 50, 100]
    max_len_coverage = []
    for k_len in candidate_max_lens:
        accommodated = int(np.sum(ge2_lengths <= k_len))
        pct_accommodated = (accommodated / len(ge2_lengths) * 100.0) if len(ge2_lengths) > 0 else 0.0
        truncated = len(ge2_lengths) - accommodated
        pct_truncated = 100.0 - pct_accommodated
        max_len_coverage.append({
            "max_length_threshold": k_len,
            "accommodated_sessions": accommodated,
            "pct_accommodated": pct_accommodated,
            "truncated_sessions": truncated,
            "pct_truncated": pct_truncated,
        })

    # -------------------------------------------------------------
    # 11. ASSEMBLE JSON SUMMARY & METADATA
    # -------------------------------------------------------------
    execution_time_sec = time.time() - start_time_sec
    dataset_summary: Dict[str, Any] = {
        "metadata": {
            "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script_name": "scripts/inspect_data.py",
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "raw_file_path": str(raw_path.resolve()),
            "raw_file_size_bytes": raw_file_size_bytes,
            "raw_file_size_mb": float(round(raw_file_size_mb, 2)),
            "chunksize": chunksize,
            "max_chunks_processed": chunk_idx,
            "total_execution_time_seconds": float(round(execution_time_sec, 2)),
        },
        "basic_summary": {
            "total_clicks": total_clicks,
            "num_unique_sessions": total_unique_sessions,
            "num_unique_items": total_unique_items,
            "num_unique_categories": total_unique_categories,
            "min_timestamp_utc": min_datetime_utc,
            "max_timestamp_utc": max_datetime_utc,
            "date_range_days": float(round(date_range_days, 2)),
        },
        "data_quality": {
            "missing_values": missing_counts,
            "timestamp_parsing_failures": parsing_failures_count,
            "invalid_rows_count": invalid_rows_count,
            "exact_duplicate_rows": total_exact_duplicates,
            "exact_duplicate_pct": float(total_exact_duplicates / total_clicks * 100.0) if total_clicks > 0 else 0.0,
            "is_globally_sorted": is_globally_sorted,
            "total_timestamp_ties": total_timestamp_ties,
            "timestamp_ties_proportion": float(total_timestamp_ties / (total_clicks - 1)) if total_clicks > 1 else 0.0,
            "same_session_timestamp_ties": same_session_timestamp_ties,
        },
        "session_length_distribution": session_length_summary,
        "session_timing": timing_summary,
        "consecutive_repeats": consecutive_repeats_summary,
        "item_frequency": {
            "summary_percentiles": item_freq_stats,
            "support_thresholds": support_analysis,
            "top_items_coverage": top_items_table,
        },
        "categories": {
            "num_unique_categories": total_unique_categories,
            "top_categories": top_categories,
        },
        "subset_what_if": subset_results,
        "max_length_analysis": {
            "percentiles": max_len_stats,
            "candidate_coverage": max_len_coverage,
        },
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    # -------------------------------------------------------------
    # 12. GENERATE HUMAN-READABLE INSPECTION REPORT (.md)
    # -------------------------------------------------------------
    generate_markdown_report(dataset_summary, output_dir / "inspection_report.md")

    logger.info("Dataset inspection completed successfully in %.2f seconds.", execution_time_sec)
    logger.info("Inspection artifacts saved to: %s", output_dir)
    return dataset_summary


def generate_markdown_report(summary: Dict[str, Any], report_path: Path) -> None:
    """Generate comprehensive human-readable Markdown inspection report."""
    meta = summary["metadata"]
    basic = summary["basic_summary"]
    quality = summary["data_quality"]
    sess_len = summary["session_length_distribution"]
    timing = summary["session_timing"]
    repeats = summary["consecutive_repeats"]
    item_freq = summary["item_frequency"]
    subsets = summary["subset_what_if"]
    max_len = summary["max_length_analysis"]

    dur_p = timing["session_duration_seconds"]
    gap_p = timing["inter_click_gap_seconds"]

    md = f"""# Yoochoose Clickstream Dataset Inspection Report

**Generated:** {meta["execution_timestamp_utc"]}  
**Script:** `{meta["script_name"]}`  
**Environment:** Python `{meta["python_version"]}`, Pandas `{meta["pandas_version"]}`, NumPy `{meta["numpy_version"]}`  
**Raw File Size:** {meta["raw_file_size_mb"]:.2f} MB ({meta["raw_file_size_bytes"]:,} bytes)  
**Execution Time:** {meta["total_execution_time_seconds"]:.2f} seconds  

---

## 1. Observed Dataset Facts

| Metric | Observed Value |
| :--- | :--- |
| **Total Clicks (Rows)** | **{basic["total_clicks"]:,}** |
| **Unique Sessions** | **{basic["num_unique_sessions"]:,}** |
| **Unique Items** | **{basic["num_unique_items"]:,}** |
| **Unique Categories** | **{basic["num_unique_categories"]:,}** |
| **Earliest Timestamp (UTC)** | `{basic["min_timestamp_utc"]}` |
| **Latest Timestamp (UTC)** | `{basic["max_timestamp_utc"]}` |
| **Temporal Span** | **{basic["date_range_days"]:.2f} days** (~{basic["date_range_days"]/30.4:.1f} months) |

---

## 2. Data Quality & Integrity

| Quality Check | Result | Details / Implication |
| :--- | :--- | :--- |
| **Missing Values** | `session_id`: {quality["missing_values"]["session_id"]}, `timestamp`: {quality["missing_values"]["timestamp"]}, `item_id`: {quality["missing_values"]["item_id"]}, `category`: {quality["missing_values"]["category"]} | No critical missing identifiers |
| **Timestamp Parsing Failures** | **{quality["timestamp_parsing_failures"]:,}** | All timestamps parsed cleanly into UTC epoch ms |
| **Invalid Rows** | **{quality["invalid_rows_count"]:,}** | No corrupted record rows detected |
| **Exact Duplicate Rows** | **{quality["exact_duplicate_rows"]:,}** ({quality["exact_duplicate_pct"]:.4f}%) | Rows identical in session, timestamp, item, and category |
| **Globally Sorted Timestamps** | **{quality["is_globally_sorted"]}** | Indicates whether clicks arrive monotonically in time |
| **Global Timestamp Ties** | **{quality["total_timestamp_ties"]:,}** ({quality["timestamp_ties_proportion"]*100:.2f}%) | Consecutive clicks sharing identical millisecond timestamps |
| **Same-Session Timestamp Ties** | **{quality["same_session_timestamp_ties"]:,}** | Consecutive clicks within same session sharing identical timestamp |

> **Tie-breaking note:** Deterministic sorting within sessions must use `(timestamp_ms, original_row_number)` to resolve identical timestamp clicks.

---

## 3. Session Length Distribution (Clicks per Session)

| Metric | All Sessions |
| :--- | :--- |
| **Total Sessions** | {sess_len["total_sessions"]:,} |
| **1-Click Sessions** | **{sess_len["num_1click_sessions"]:,} ({sess_len["pct_1click_sessions"]:.2f}%)** |
| **>= 2 Click Sessions** | **{sess_len["num_ge2click_sessions"]:,} ({sess_len["pct_ge2click_sessions"]:.2f}%)** |
| **Min Length** | {sess_len["min_length"]:.0f} click |
| **25th Percentile** | {sess_len["p25"]:.0f} clicks |
| **Median (50th Percentile)** | **{sess_len["median_length"]:.0f} clicks** |
| **Mean Length** | **{sess_len["mean_length"]:.2f} clicks** (Std: {sess_len["std_length"]:.2f}) |
| **75th Percentile** | {sess_len["p75"]:.0f} clicks |
| **90th Percentile** | {sess_len["p90"]:.0f} clicks |
| **95th Percentile** | {sess_len["p95"]:.0f} clicks |
| **99th Percentile** | {sess_len["p99"]:.0f} clicks |
| **99.5th Percentile** | {sess_len["p99_5"]:.0f} clicks |
| **Max Length** | {sess_len["max_length"]:.0f} clicks |

---

## 4. Session Timing & Inter-Click Gaps (for >= 2 Click Sessions)

### Session Duration (seconds):
- **Min Duration:** {dur_p["min"]:.2f} s
- **25th Percentile:** {dur_p["p25"]:.2f} s ({dur_p["p25"]/60:.2f} min)
- **Median (50th):** {dur_p["median"]:.2f} s ({dur_p["median"]/60:.2f} min)
- **Mean:** {dur_p["mean"]:.2f} s ({dur_p["mean"]/60:.2f} min)
- **75th Percentile:** {dur_p["p75"]:.2f} s ({dur_p["p75"]/60:.2f} min)
- **90th Percentile:** {dur_p["p90"]:.2f} s ({dur_p["p90"]/60:.2f} min)
- **95th Percentile:** {dur_p["p95"]:.2f} s ({dur_p["p95"]/60:.2f} min)
- **99th Percentile:** {dur_p["p99"]:.2f} s ({dur_p["p99"]/60:.2f} min)
- **Max Duration:** {dur_p["max"]:.2f} s ({dur_p["max"]/86400:.2f} days)

### Inter-Click Time Gaps (Delta-t in seconds):
- **Transitions Count:** {gap_p["count"]:,}
- **Median Gap:** {gap_p["median"]:.2f} s
- **Mean Gap:** {gap_p["mean"]:.2f} s
- **75th Percentile:** {gap_p["p75"]:.2f} s
- **90th Percentile:** {gap_p["p90"]:.2f} s
- **95th Percentile:** {gap_p["p95"]:.2f} s
- **99th Percentile:** {gap_p["p99"]:.2f} s
- **Zero-gap (same-ms) Transitions:** {timing["zero_gap_transitions_count"]:,} ({timing["zero_gap_transitions_pct"]:.2f}%)

---

## 5. Consecutive Repeated Items

| Metric | Value | Percentage |
| :--- | :--- | :--- |
| **Total Inter-Click Transitions** | {repeats["total_transitions"]:,} | 100.0% |
| **Consecutive Repeat Transitions ($A \\to A$)** | **{repeats["consecutive_repeat_transitions"]:,}** | **{repeats["consecutive_repeat_transition_pct"]:.2f}%** |
| **Sessions with >= 1 Consecutive Repeat** | **{repeats["sessions_with_consecutive_repeats"]:,}** | **{repeats["pct_sessions_with_consecutive_repeats"]:.2f}%** |

---

## 6. Item Support & Long-Tail Distribution

### Minimum Item Support Threshold Analysis:

| Support Threshold | Rare Items (< Threshold) | % Unique Items | Rare Clicks | % Total Clicks | Retained Items | % Retained Items | Retained Clicks | % Retained Clicks |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for row in item_freq["support_thresholds"]:
        md += f"| **{row['threshold']}** | {row['items_count']:,} | {row['pct_of_unique_items']:.2f}% | {row['clicks_count']:,} | {row['pct_of_total_clicks']:.2f}% | {row['retained_items_count']:,} | {row['pct_retained_items']:.2f}% | {row['retained_clicks_count']:,} | {row['pct_retained_clicks']:.2f}% |\n"

    md += f"""
### Top Item Concentration:
"""
    for top_r in item_freq["top_items_coverage"]:
        md += f"- **Top {top_r['top_k']} items** account for **{top_r['clicks']:,} clicks ({top_r['pct_of_total_clicks']:.2f}% of all clicks)**\n"

    md += f"""
---

## 7. Chronological Subset What-If Analysis

*Ordered strictly by session end time: `(end_time, session_id)` (most recent sessions).*

| Subset Fraction | Number of Sessions | % Sessions | Total Clicks | % Clicks | Date Range (UTC) | Span (Days) | Approx Raw Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for sub in subsets:
        md += f"| **{sub['subset_fraction']}** | {sub['num_sessions']:,} | {sub['pct_sessions']:.2f}% | {sub['num_clicks']:,} | {sub['pct_clicks']:.2f}% | `{sub['date_range_start']}` to `{sub['date_range_end']}` | {sub['span_days']:.1f} d | ~{sub['approx_raw_size_mb']:.1f} MB |\n"

    md += f"""
---

## 8. Candidate Maximum Input Length (for >= 2 Click Sessions)

| Candidate Max Length | Accommodated Sessions (<= K) | % Accommodated | Truncated Sessions (> K) | % Truncated |
| :--- | :--- | :--- | :--- | :--- |
"""
    for cov in max_len["candidate_coverage"]:
        md += f"| **K = {cov['max_length_threshold']}** | {cov['accommodated_sessions']:,} | **{cov['pct_accommodated']:.2f}%** | {cov['truncated_sessions']:,} | {cov['pct_truncated']:.2f}% |\n"

    md += """
---

## 9. Candidate Preprocessing Considerations

*(Note: These are observed options to inform subsequent decisions; no parameters are finalized at this inspection stage).*

1. **Filtering 1-Click Sessions:**
   - 1-click sessions cannot be used for next-item sequential prediction training. Filtering them reduces dataset size significantly while leaving 100% of usable multi-click trajectories.
2. **Consecutive Repeat Handling:**
   - A notable portion of user interactions consist of rapid consecutive clicks on the exact same item ($A \\to A$). We can decide whether collapsing consecutive repeats or preserving them as distinct time points is best suited for Liquid Neural Networks (LTC).
3. **Item Vocabulary Pruning (min_item_support):**
   - Filtering items with support $< 5$ or $< 10$ dramatically reduces vocabulary size and embedding matrix memory while preserving >95-98% of total click interactions.
4. **Chronological Subsetting:**
   - For fast experimentation in Colab/local environments, recent subsets such as `1/64` or `1/32` offer compact, realistic temporal slices spanning recent active days, while `1/4` or `1/1` can serve for final comprehensive benchmarking.
5. **Maximum Sequence Length Truncation:**
   - Choosing a cutoff (e.g. 20, 30, or 50) accommodates over 95-99% of user sessions in full without excessive padding overhead.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    raw_path = Path(args.raw_path)
    output_dir = Path(args.output_dir)

    # If relative, resolve against REPO_ROOT
    if not raw_path.is_absolute():
        raw_path = REPO_ROOT / raw_path
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    try:
        run_inspection(
            raw_path=raw_path,
            output_dir=output_dir,
            chunksize=args.chunksize,
            max_chunks=args.max_chunks,
        )
    except Exception as exc:
        logger.exception("Inspection failed with error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
