#!/usr/bin/env python3
"""
Preprocessing Pipeline for Yoochoose Clickstream Dataset.

Implements the finalized deterministic preprocessing pipeline:
- Chunked raw loading (P1)
- Global stable sorting (P2)
- Raw session metadata extraction (P3)
- Exact duplicate removal (P4)
- Consecutive repeat collapse (P5)
- Session length eligibility filtering (P6)
- Chronological subsetting by (raw_end_ms, session_id) (P7)
- Chronological train/val/test split (P8)
- Iterative train-only vocabulary pruning (P9)
- Out-of-vocabulary evaluation split cleaning (P10)
- Item encoding and vocabulary serialization (P11)
- Vectorized example generation (P12)
- Full artifact serialization, hashing, and V1-V14 validation (P13)
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import io
import json
import logging
import math
from pathlib import Path
import platform
import shutil
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
    apply_chronological_subset,
    build_raw_session_table,
    clean_eval_split,
    collapse_repeats_arrays,
    compute_array_fingerprint,
    compute_directory_fingerprints,
    encode_splits,
    filter_eligible_sessions,
    generate_examples_vectorized,
    get_git_info,
    global_stable_sort,
    hash_file_sha256,
    load_raw_clicks_pipeline,
    prune_train_vocabulary,
    remove_exact_duplicates,
    run_validation_checks,
    split_chronological,
)

# Configure logging
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def parse_fraction(val: str) -> Fraction:
    """Parse a fraction string into an exact Fraction object."""
    try:
        f = Fraction(val)
        if f <= 0 or f > 1:
            raise ValueError(f"Fraction must be in range (0, 1], got {val}")
        return f
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid fraction '{val}': {e}")


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deterministic, memory-efficient preprocessing for Yoochoose clicks."
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/yoochoose_clicks.dat",
        help="Path to raw yoochoose_clicks.dat (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/processed",
        help="Root directory for processed datasets.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/preprocessing",
        help="Root directory for preprocessing logs and reports.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=2_000_000,
        help="Chunksize for streaming raw read (default: 2,000,000).",
    )
    parser.add_argument(
        "--subset-fraction",
        type=parse_fraction,
        default=Fraction(1, 64),
        help="Fraction of most recent sessions to retain (default: 1/64).",
    )
    parser.add_argument(
        "--min-session-length",
        type=int,
        default=2,
        help="Minimum session length in clicks (must be >= 2, default: 2).",
    )
    parser.add_argument(
        "--min-item-support",
        type=int,
        default=5,
        help="Minimum train click count for vocabulary items (default: 5).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=20,
        help="Maximum input sequence length (default: 20).",
    )
    parser.add_argument(
        "--val-fraction",
        type=parse_fraction,
        default=Fraction(1, 10),
        help="Fraction of sessions for validation split (default: 0.1).",
    )
    parser.add_argument(
        "--test-fraction",
        type=parse_fraction,
        default=Fraction(1, 10),
        help="Fraction of sessions for test split (default: 0.1).",
    )
    parser.add_argument(
        "--collapse-consecutive-repeats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collapse consecutive repeated items within sessions (default: True).",
    )
    parser.add_argument(
        "--save-time-deltas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute and save input_delta_t arrays in seconds (default: True).",
    )
    parser.add_argument(
        "--hash-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute SHA-256 hash of raw file (default: True, skipped in debug mode).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run folder after validation succeeds.",
    )
    parser.add_argument(
        "--verify-only",
        type=str,
        default=None,
        metavar="RUN_NAME",
        help="Verify an existing run folder without preprocessing.",
    )
    parser.add_argument(
        "--debug-max-rows",
        type=int,
        default=None,
        help="Smoke test mode: read only first N rows and output to _debug/ folder.",
    )
    return parser.parse_args(argv)


def compute_run_name(
    subset_fraction: Fraction,
    min_session_length: int,
    min_item_support: int,
    max_len: int,
    collapse_repeats: bool,
    val_fraction: Fraction,
    test_fraction: Fraction,
    save_time_deltas: bool,
) -> str:
    """Generate canonical run name."""
    f_str = "full" if subset_fraction == Fraction(1, 1) else f"{subset_fraction.numerator}-{subset_fraction.denominator}"
    train_frac = 1 - val_fraction - test_fraction
    tr_str = f"{100 * float(train_frac):g}"
    va_str = f"{100 * float(val_fraction):g}"
    te_str = f"{100 * float(test_fraction):g}"
    c_str = "collapse" if collapse_repeats else "nocollapse"
    base_name = f"frac_{f_str}_minlen{min_session_length}_minsup{min_item_support}_maxlen{max_len}_{c_str}_split{tr_str}-{va_str}-{te_str}"
    if not save_time_deltas:
        base_name += "_nodt"
    return base_name


def format_iso(ts_ms: int) -> str:
    """Format epoch millisecond timestamp to ISO UTC string."""
    return pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def compute_time_delta_stats(train_sess_timestamps: np.ndarray, train_offsets: np.ndarray) -> Dict[str, Any]:
    """Compute time delta distribution and log1p statistics on train transitions."""
    all_deltas = []
    for i in range(len(train_offsets) - 1):
        ts = train_sess_timestamps[train_offsets[i] : train_offsets[i + 1]]
        if len(ts) > 1:
            diffs = (ts[1:] - ts[:-1]) / 1000.0
            all_deltas.append(diffs.astype(np.float64))

    if len(all_deltas) == 0:
        return {
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p99_9": 0.0,
            "max": 0.0,
            "log1p_mean": 0.0,
            "log1p_std": 0.0,
        }

    deltas_arr = np.concatenate(all_deltas)
    pcts = np.percentile(deltas_arr, [50, 90, 95, 99, 99.9])
    log1p_arr = np.log1p(deltas_arr)

    return {
        "p50": float(pcts[0]),
        "p90": float(pcts[1]),
        "p95": float(pcts[2]),
        "p99": float(pcts[3]),
        "p99_9": float(pcts[4]),
        "max": float(np.max(deltas_arr)),
        "log1p_mean": float(np.mean(log1p_arr)),
        "log1p_std": float(np.std(log1p_arr)),
    }


def compute_session_length_stats(offsets: np.ndarray) -> Dict[str, Any]:
    """Compute summary statistics for session lengths."""
    lens = offsets[1:] - offsets[:-1]
    if len(lens) == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    pcts = np.percentile(lens, [50, 90, 99])
    return {
        "mean": float(np.mean(lens)),
        "median": float(pcts[0]),
        "p90": float(pcts[1]),
        "p99": float(pcts[2]),
        "max": float(np.max(lens)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Main execution function for preprocessing script."""
    start_time = time.time()
    args = parse_arguments(argv)

    # Resolve paths relative to REPO_ROOT
    raw_path = Path(args.raw_path)
    if not raw_path.is_absolute():
        raw_path = REPO_ROOT / raw_path

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = REPO_ROOT / results_root

    # Verify parameters
    if args.min_session_length < 2:
        logger.error("min-session-length must be >= 2, got %d", args.min_session_length)
        return 1
    if args.val_fraction + args.test_fraction >= 1:
        logger.error("val-fraction + test-fraction must be < 1, got %s + %s", args.val_fraction, args.test_fraction)
        return 1

    is_debug = args.debug_max_rows is not None
    if is_debug:
        logger.warning(
            "==================================================================\n"
            "RUNNING IN DEBUG MODE WITH --debug-max-rows %d\n"
            "Outputs will be written to _debug/ subfolders.\n"
            "==================================================================",
            args.debug_max_rows,
        )

    # -------------------------------------------------------------
    # VERIFY-ONLY MODE
    # -------------------------------------------------------------
    if args.verify_only:
        run_name = args.verify_only
        target_dir = (output_root / "_debug" / run_name) if is_debug else (output_root / run_name)
        if not target_dir.exists():
            # Check without debug prefix
            target_dir = output_root / run_name
            if not target_dir.exists():
                logger.error("Run directory not found for verification: %s", target_dir)
                return 1

        logger.info("Verifying existing run folder: %s", target_dir)
        meta_file = target_dir / "metadata.json"
        if not meta_file.exists():
            logger.error("metadata.json missing in %s", target_dir)
            return 1
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_stat = (meta["raw_file"]["size_bytes"], meta["raw_file"]["mtime_ns"]) if raw_path.exists() else (0, 0)
        passed, v_results = run_validation_checks(
            data_dir=target_dir,
            raw_path=raw_path,
            raw_initial_stat=raw_stat,
            params={
                "max_len": meta["resolved_args"]["max_len"],
                "min_session_length": meta["resolved_args"]["min_session_length"],
                "min_item_support": meta["resolved_args"]["min_item_support"],
                "save_time_deltas": meta["resolved_args"]["save_time_deltas"],
                "collapse_consecutive_repeats": meta["resolved_args"]["collapse_consecutive_repeats"],
            },
            in_memory_fingerprints=meta.get("fingerprints"),
            is_debug=is_debug,
        )
        if passed:
            logger.info("Verification PASSED for run %s", run_name)
            return 0
        else:
            logger.error("Verification FAILED for run %s", run_name)
            return 1

    # -------------------------------------------------------------
    # PREPROCESSING EXECUTION
    # -------------------------------------------------------------
    run_name = compute_run_name(
        subset_fraction=args.subset_fraction,
        min_session_length=args.min_session_length,
        min_item_support=args.min_item_support,
        max_len=args.max_len,
        collapse_repeats=args.collapse_consecutive_repeats,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        save_time_deltas=args.save_time_deltas,
    )

    if is_debug:
        final_output_dir = output_root / "_debug" / run_name
        final_results_dir = results_root / "_debug" / run_name
    else:
        final_output_dir = output_root / run_name
        final_results_dir = results_root / run_name

    if final_output_dir.exists() and not args.overwrite:
        logger.error(
            "Output directory already exists: %s\n"
            "Use --overwrite to re-run and replace after validation.",
            final_output_dir,
        )
        return 1

    partial_output_dir = final_output_dir.parent / f"{final_output_dir.name}.partial"
    if partial_output_dir.exists():
        shutil.rmtree(partial_output_dir)
    partial_output_dir.mkdir(parents=True, exist_ok=True)
    final_results_dir.mkdir(parents=True, exist_ok=True)

    # Capture logs to file
    log_stream = io.StringIO()
    file_log_handler = logging.StreamHandler(log_stream)
    file_log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(file_log_handler)

    # Record raw file initial stat
    if not raw_path.exists():
        logger.error(
            "Raw dataset file not found at %s.\n"
            "If running in Google Colab, copy the dataset from Google Drive to data/raw/yoochoose_clicks.dat.",
            raw_path,
        )
        return 1

    raw_st = raw_path.stat()
    raw_initial_stat = (raw_st.st_size, raw_st.st_mtime_ns)
    raw_sha256 = None
    if args.hash_raw and not is_debug:
        logger.info("Computing SHA-256 for raw file (this may take a minute)...")
        raw_sha256 = hash_file_sha256(raw_path)
        logger.info("Raw file SHA-256: %s", raw_sha256)
    elif is_debug:
        logger.info("Skipping raw file hash in debug mode.")

    drop_ledger: List[Dict[str, Any]] = []

    try:
        # P1: Load
        logger.info("P1: Loading raw clicks (chunksize=%d, max_rows=%s)...", args.chunksize, args.debug_max_rows)
        session_ids, timestamps_ms, item_ids, row_indices, total_rows_read = load_raw_clicks_pipeline(
            raw_path=raw_path,
            chunksize=args.chunksize,
            max_rows=args.debug_max_rows,
        )
        drop_ledger.append({
            "step": "P1_load",
            "unit": "rows_read",
            "count": total_rows_read,
            "note": "Total raw rows read from file",
        })

        # P2: Global Stable Sort
        logger.info("P2: Performing global stable sort by (session_id, timestamp_ms, row_indices)...")
        session_ids, timestamps_ms, item_ids, row_indices = global_stable_sort(
            session_ids, timestamps_ms, item_ids, row_indices
        )

        # P3: Raw Session Table
        logger.info("P3: Building raw session table before cleaning...")
        raw_u_sids, raw_counts, raw_ends, raw_session_table = build_raw_session_table(
            session_ids, timestamps_ms
        )
        drop_ledger.append({
            "step": "P3_raw_sessions",
            "unit": "sessions",
            "count": len(raw_u_sids),
            "note": "Total distinct sessions in raw data",
        })

        # P4: Exact Duplicates
        logger.info("P4: Detecting and removing exact duplicate rows...")
        session_ids, timestamps_ms, item_ids, row_indices, num_dups_removed = remove_exact_duplicates(
            session_ids, timestamps_ms, item_ids, row_indices
        )
        drop_ledger.append({
            "step": "P4_exact_duplicates",
            "unit": "clicks",
            "count": num_dups_removed,
            "note": "Exact duplicate rows removed",
        })

        # P5: Collapse Consecutive Repeats
        num_initial_repeats_collapsed = 0
        if args.collapse_consecutive_repeats:
            logger.info("P5: Collapsing consecutive repeated items within sessions...")
            session_ids, timestamps_ms, item_ids, row_indices, num_initial_repeats_collapsed = collapse_repeats_arrays(
                session_ids, timestamps_ms, item_ids, row_indices
            )
            drop_ledger.append({
                "step": "P5_consecutive_repeats",
                "unit": "clicks",
                "count": num_initial_repeats_collapsed,
                "note": "Consecutive repeated items collapsed",
            })

        # P6: Eligibility
        logger.info("P6: Filtering sessions with cleaned length >= %d...", args.min_session_length)
        session_ids, timestamps_ms, item_ids, row_indices, eligible_sids, elig_drop_info = filter_eligible_sessions(
            session_ids, timestamps_ms, item_ids, row_indices, args.min_session_length, raw_session_table
        )
        drop_ledger.append({
            "step": "P6_dropped_1click_sessions",
            "unit": "sessions",
            "count": elig_drop_info["dropped_1click_sessions"],
            "clicks_dropped": elig_drop_info["dropped_1click_clicks"],
            "note": "Sessions with 1 raw click dropped",
        })
        drop_ledger.append({
            "step": "P6_dropped_reduced_sessions",
            "unit": "sessions",
            "count": elig_drop_info["dropped_reduced_sessions"],
            "clicks_dropped": elig_drop_info["dropped_reduced_clicks"],
            "note": "Sessions reduced below min_session_length by cleaning dropped",
        })

        # P7: Subset
        logger.info("P7: Applying chronological subset fraction %s...", args.subset_fraction)
        session_ids, timestamps_ms, item_ids, row_indices, kept_sids, subset_meta = apply_chronological_subset(
            session_ids, timestamps_ms, item_ids, row_indices, eligible_sids, raw_session_table, args.subset_fraction
        )
        drop_ledger.append({
            "step": "P7_subset_dropped",
            "unit": "sessions",
            "count": subset_meta["dropped_sessions"],
            "clicks_dropped": subset_meta["dropped_clicks"],
            "note": "Sessions outside chronological subset dropped",
        })

        # P8: Chronological Split
        logger.info("P8: Splitting chronological sessions into train/val/test...")
        splits, split_meta = split_chronological(
            session_ids, timestamps_ms, item_ids, kept_sids, raw_session_table, args.val_fraction, args.test_fraction
        )

        # P9: Train Vocabulary Pruning
        logger.info("P9: Iteratively pruning train vocabulary with min_support=%d...", args.min_item_support)
        vocabulary, cleaned_train, train_iter_logs = prune_train_vocabulary(
            splits["train"], args.min_item_support, args.min_session_length, args.collapse_consecutive_repeats
        )
        for log_entry in train_iter_logs:
            drop_ledger.append({
                "step": f"P9_train_prune_iter_{log_entry['iteration']}",
                "unit": "items_and_clicks",
                "items_removed": log_entry["rare_items_removed"],
                "clicks_removed": log_entry["rare_clicks_removed"],
                "repeats_collapsed": log_entry["repeats_collapsed"],
                "sessions_dropped": log_entry["sessions_dropped"],
                "clicks_in_dropped_sessions": log_entry["clicks_in_dropped_sessions"],
                "note": f"Train vocabulary pruning iteration {log_entry['iteration']}",
            })

        # P10: Val/Test Cleaning
        logger.info("P10: Cleaning val and test splits against train vocabulary...")
        cleaned_val, val_drop_stats = clean_eval_split(
            splits["val"], vocabulary, args.min_session_length, args.collapse_consecutive_repeats, "val"
        )
        cleaned_test, test_drop_stats = clean_eval_split(
            splits["test"], vocabulary, args.min_session_length, args.collapse_consecutive_repeats, "test"
        )
        drop_ledger.append({"step": "P10_val_cleaning", **val_drop_stats})
        drop_ledger.append({"step": "P10_test_cleaning", **test_drop_stats})

        # P11: Encode
        logger.info("P11: Encoding items (vocab size=%d, PAD=0)...", len(vocabulary) + 1)
        encoded_splits, item_index_df, item2idx, sorted_vocab = encode_splits(
            cleaned_train, cleaned_val, cleaned_test, vocabulary
        )

        # P12: Example Generation
        logger.info("P12: Vectorized example generation (max_len=%d, save_deltas=%s)...", args.max_len, args.save_time_deltas)
        train_ex, train_sess = generate_examples_vectorized(
            encoded_splits["train"], raw_session_table, args.max_len, args.save_time_deltas
        )
        val_ex, val_sess = generate_examples_vectorized(
            encoded_splits["val"], raw_session_table, args.max_len, args.save_time_deltas
        )
        test_ex, test_sess = generate_examples_vectorized(
            encoded_splits["test"], raw_session_table, args.max_len, args.save_time_deltas
        )

        # P13: Save to partial directory
        logger.info("P13: Saving output files to %s...", partial_output_dir)
        np.savez_compressed(partial_output_dir / "train.npz", **train_ex)
        np.savez_compressed(partial_output_dir / "val.npz", **val_ex)
        np.savez_compressed(partial_output_dir / "test.npz", **test_ex)

        np.savez_compressed(partial_output_dir / "sessions_train.npz", **train_sess)
        np.savez_compressed(partial_output_dir / "sessions_val.npz", **val_sess)
        np.savez_compressed(partial_output_dir / "sessions_test.npz", **test_sess)

        item_index_df.to_csv(partial_output_dir / "item_index.csv", index=False, encoding="utf-8", lineterminator="\n")
        with open(partial_output_dir / "item2idx.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(item2idx, f, indent=2)

        # Compute in-memory fingerprints
        in_mem_fingerprints = compute_directory_fingerprints(partial_output_dir)

        # Validation
        logger.info("Running V1-V14 validation suite...")
        passed, val_results = run_validation_checks(
            data_dir=partial_output_dir,
            raw_path=raw_path,
            raw_initial_stat=raw_initial_stat,
            params={
                "max_len": args.max_len,
                "min_session_length": args.min_session_length,
                "min_item_support": args.min_item_support,
                "save_time_deltas": args.save_time_deltas,
                "collapse_consecutive_repeats": args.collapse_consecutive_repeats,
            },
            in_memory_fingerprints=in_mem_fingerprints,
            is_debug=is_debug,
            inspection_summary_path=REPO_ROOT / "results/inspection/dataset_summary.json",
        )

        # Compute stats for metadata & report
        train_offsets = train_sess["offsets"]
        val_offsets = val_sess["offsets"]
        test_offsets = test_sess["offsets"]

        train_delta_stats = compute_time_delta_stats(train_sess["timestamps_ms"], train_offsets) if args.save_time_deltas else {}
        session_len_stats = {
            "train": compute_session_length_stats(train_offsets),
            "val": compute_session_length_stats(val_offsets),
            "test": compute_session_length_stats(test_offsets),
        }

        # Date ranges per split
        def _get_dates(sess_dict: Dict[str, np.ndarray]) -> Dict[str, str]:
            if len(sess_dict["timestamps_ms"]) == 0:
                return {"min_click": "N/A", "max_click": "N/A", "min_end": "N/A", "max_end": "N/A"}
            return {
                "min_click": format_iso(int(np.min(sess_dict["timestamps_ms"]))),
                "max_click": format_iso(int(np.max(sess_dict["timestamps_ms"]))),
                "min_end": format_iso(int(np.min(sess_dict["end_time_ms"]))),
                "max_end": format_iso(int(np.max(sess_dict["end_time_ms"]))),
            }

        split_date_info = {
            "train": _get_dates(train_sess),
            "val": _get_dates(val_sess),
            "test": _get_dates(test_sess),
        }

        git_info = get_git_info(REPO_ROOT)
        runtime_sec = time.time() - start_time

        # Metadata dictionary
        metadata = {
            "schema_version": "1.0",
            "run_name": run_name,
            "command_line": " ".join(sys.argv),
            "resolved_args": {
                "raw_path": str(raw_path.relative_to(REPO_ROOT) if raw_path.is_relative_to(REPO_ROOT) else raw_path.name),
                "output_root": str(output_root.relative_to(REPO_ROOT) if output_root.is_relative_to(REPO_ROOT) else output_root.name),
                "results_root": str(results_root.relative_to(REPO_ROOT) if results_root.is_relative_to(REPO_ROOT) else results_root.name),
                "chunksize": args.chunksize,
                "subset_fraction": str(args.subset_fraction),
                "min_session_length": args.min_session_length,
                "min_item_support": args.min_item_support,
                "max_len": args.max_len,
                "val_fraction": str(args.val_fraction),
                "test_fraction": str(args.test_fraction),
                "collapse_consecutive_repeats": args.collapse_consecutive_repeats,
                "save_time_deltas": args.save_time_deltas,
                "debug_max_rows": args.debug_max_rows,
            },
            "raw_file": {
                "file_name": raw_path.name,
                "size_bytes": raw_initial_stat[0],
                "mtime_ns": raw_initial_stat[1],
                "sha256": raw_sha256,
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "platform": platform.platform(),
                "git_commit": git_info["git_commit"],
                "git_dirty": git_info["git_dirty"],
            },
            "drop_ledger": drop_ledger,
            "subset": subset_meta,
            "split": {
                "train_sessions": len(train_sess["session_ids"]),
                "train_clicks": len(train_sess["items"]),
                "train_examples": len(train_ex["targets"]),
                "val_sessions": len(val_sess["session_ids"]),
                "val_clicks": len(val_sess["items"]),
                "val_examples": len(val_ex["targets"]),
                "test_sessions": len(test_sess["session_ids"]),
                "test_clicks": len(test_sess["items"]),
                "test_examples": len(test_ex["targets"]),
                "date_ranges": split_date_info,
                "cutoffs": {
                    "train_cutoff_iso": split_meta["train_cutoff_iso"],
                    "val_cutoff_iso": split_meta["val_cutoff_iso"],
                },
            },
            "vocabulary": {
                "pad_index": 0,
                "num_items": len(vocabulary),
                "vocab_size": len(vocabulary) + 1,
                "min_support": args.min_item_support,
                "final_min_count": int(item_index_df["train_click_count"].min()) if len(item_index_df) > 0 else 0,
                "iterations": len(train_iter_logs),
            },
            "oov": {
                "val": val_drop_stats,
                "test": test_drop_stats,
            },
            "session_length_stats": session_len_stats,
            "time_delta_stats": train_delta_stats,
            "validation": val_results,
            "validation_passed": passed,
            "fingerprints": in_mem_fingerprints,
            "runtime_seconds": float(round(runtime_sec, 2)),
            "peak_rss_mb": None,
        }

        # Write metadata.json
        with open(partial_output_dir / "metadata.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(metadata, f, indent=2)

        # Write preprocessing_report.md
        report_md = generate_preprocessing_report(metadata)
        with open(partial_output_dir / "preprocessing_report.md", "w", encoding="utf-8", newline="\n") as f:
            f.write(report_md)

        # Check validation result
        if not passed:
            logger.error("Validation failed. Keeping partial run directory for debugging: %s", partial_output_dir)
            return 1

        # Move partial to final destination
        if final_output_dir.exists():
            shutil.rmtree(final_output_dir)
        partial_output_dir.rename(final_output_dir)

        # Write _SUCCESS
        (final_output_dir / "_SUCCESS").touch()

        # Copy report, metadata, and log to results_root
        shutil.copy2(final_output_dir / "metadata.json", final_results_dir / "metadata.json")
        shutil.copy2(final_output_dir / "preprocessing_report.md", final_results_dir / "preprocessing_report.md")

        with open(final_results_dir / "preprocessing.log", "w", encoding="utf-8", newline="\n") as f:
            f.write(log_stream.getvalue())

        logger.info("Preprocessing run %s completed successfully in %.2f seconds.", run_name, runtime_sec)
        logger.info("Processed output directory: %s", final_output_dir)
        logger.info("Results directory: %s", final_results_dir)
        return 0

    except Exception as e:
        logger.exception("Preprocessing pipeline encountered an unhandled exception: %s", e)
        if partial_output_dir.exists():
            fail_meta = {
                "run_name": run_name,
                "validation_passed": False,
                "error": str(e),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            with open(partial_output_dir / "metadata.json", "w", encoding="utf-8", newline="\n") as f:
                json.dump(fail_meta, f, indent=2)
        return 1


def generate_preprocessing_report(meta: Dict[str, Any]) -> str:
    """Generate human-readable markdown report for the preprocessing run."""
    args = meta["resolved_args"]
    raw = meta["raw_file"]
    sub = meta["subset"]
    sp = meta["split"]
    voc = meta["vocabulary"]
    val = meta["validation"]
    td = meta["time_delta_stats"]
    fp = meta["fingerprints"]

    md = f"""# Preprocessing Report: `{meta["run_name"]}`

**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}  
**Command Line:** `{meta["command_line"]}`  
**Environment:** Python `{meta["environment"]["python"]}`, NumPy `{meta["environment"]["numpy"]}`, Pandas `{meta["environment"]["pandas"]}`  
**Git Commit:** `{meta["environment"]["git_commit"]}` (Dirty: `{meta["environment"]["git_dirty"]}`)  
**Total Runtime:** {meta["runtime_seconds"]:.2f} seconds  
**Validation Status:** **{'PASSED' if meta['validation_passed'] else 'FAILED'}**  
**Combined Artifact Fingerprint:** `{fp.get('__combined__', 'N/A')}`  

---

## 1. Resolved Preprocessing Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Raw File** | `{raw["file_name"]}` ({raw["size_bytes"]:,} bytes) | Input clickstream dataset |
| **Subset Fraction** | **`{args["subset_fraction"]}`** | Most recent fraction of eligible sessions |
| **Min Session Length** | **`{args["min_session_length"]}`** clicks | Minimum clicks per session |
| **Min Item Support** | **`{args["min_item_support"]}`** clicks | Train-only minimum item frequency |
| **Max Sequence Length** | **`{args["max_len"]}`** items | Truncation window for input history |
| **Split Fractions** | Train: `{1 - Fraction(args["val_fraction"]) - Fraction(args["test_fraction"])}`, Val: `{args["val_fraction"]}`, Test: `{args["test_fraction"]}` | Chronological split ratios |
| **Collapse Consecutive Repeats** | **`{args["collapse_consecutive_repeats"]}`** | $A \\to A \\to B \\to A, B$ |
| **Save Time Deltas** | **`{args["save_time_deltas"]}`** | Compute $\\Delta t$ in seconds |

---

## 2. Drop Ledger & Data Filtering

| Step / Filter Stage | Unit | Count | Notes |
| :--- | :--- | :--- | :--- |
"""
    for entry in meta["drop_ledger"]:
        cnt_str = f"{entry.get('count', 0):,}" if "count" in entry else "N/A"
        md += f"| **`{entry.get('step', 'N/A')}`** | {entry.get('unit', '')} | {cnt_str} | {entry.get('note', '')} |\n"

    md += f"""
---

## 3. Chronological Dataset Splits

| Split | Sessions | Clicks | Next-Item Examples | Click Date Range (UTC) | End-Time Cutoff (UTC) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **{sp["train_sessions"]:,}** | **{sp["train_clicks"]:,}** | **{sp["train_examples"]:,}** | `{sp["date_ranges"]["train"]["min_click"]}` to `{sp["date_ranges"]["train"]["max_click"]}` | `{sp["cutoffs"]["train_cutoff_iso"]}` |
| **Validation** | **{sp["val_sessions"]:,}** | **{sp["val_clicks"]:,}** | **{sp["val_examples"]:,}** | `{sp["date_ranges"]["val"]["min_click"]}` to `{sp["date_ranges"]["val"]["max_click"]}` | `{sp["cutoffs"]["val_cutoff_iso"]}` |
| **Test** | **{sp["test_sessions"]:,}** | **{sp["test_clicks"]:,}** | **{sp["test_examples"]:,}** | `{sp["date_ranges"]["test"]["min_click"]}` to `{sp["date_ranges"]["test"]["max_click"]}` | `{sp["date_ranges"]["test"]["max_end"]}` |
| **Total Retained** | **{sp["train_sessions"] + sp["val_sessions"] + sp["test_sessions"]:,}** | **{sp["train_clicks"] + sp["val_clicks"] + sp["test_clicks"]:,}** | **{sp["train_examples"] + sp["val_examples"] + sp["test_examples"]:,}** | - | - |

---

## 4. Vocabulary & OOV Summary

- **Vocabulary Size (with PAD=0):** **{voc["vocab_size"]:,}** (Active items: {voc["num_items"]:,})
- **Minimum Train Item Count:** {voc["final_min_count"]} (Threshold: {voc["min_support"]})
- **Train Pruning Iterations:** {voc["iterations"]}
- **Validation OOV Clicks Removed:** {meta["oov"]["val"]["oov_clicks_removed"]:,} ({meta["oov"]["val"]["oov_clicks_pct"]:.2f}%)
- **Test OOV Clicks Removed:** {meta["oov"]["test"]["oov_clicks_removed"]:,} ({meta["oov"]["test"]["oov_clicks_pct"]:.2f}%)

---

## 5. Timing Statistics (Train Session Transitions)

- **Median Delta-t:** {td.get("p50", 0.0):.2f} s
- **90th Percentile:** {td.get("p90", 0.0):.2f} s
- **95th Percentile:** {td.get("p95", 0.0):.2f} s
- **99th Percentile:** {td.get("p99", 0.0):.2f} s
- **99.9th Percentile:** {td.get("p99_9", 0.0):.2f} s
- **Max Delta-t:** {td.get("max", 0.0):.2f} s
- **Log1p Delta-t:** Mean = {td.get("log1p_mean", 0.0):.4f}, Std = {td.get("log1p_std", 0.0):.4f}

---

## 6. Validation Checks Summary (V1 - V14)

| Check | Passed | Detail |
| :--- | :--- | :--- |
"""
    for v_name, v_info in val.items():
        pass_badge = "PASSED" if v_info.get("passed") else "FAILED"
        md += f"| **`{v_name}`** | {pass_badge} | {v_info.get('detail', '')} |\n"

    md += """
---

## 7. Fixed Preprocessing Notes & Caveats

1. **Out-of-Vocabulary Clicks in Val/Test:**
   - Clicks on items not present in the train vocabulary are dropped from evaluation sessions. This aligns with standard benchmark protocols (e.g. GRU4Rec), though it makes offline evaluation slightly cleaner than real-world cold-start deployment.
2. **Consecutive Repeat Collapsing:**
   - Immediate consecutive interactions on identical items ($A \\to A$) were collapsed to single interactions ($A$). Comparisons against papers that retain consecutive repeats should note this distinction.
3. **Chronological Subsets:**
   - Different subset fractions (e.g. 1/64, 1/32, 1/16, 1/8, 1/4) select nested most-recent session windows. A larger fraction contains the smaller fraction and is not an independent statistical replication.
"""
    return md


if __name__ == "__main__":
    sys.exit(main())
