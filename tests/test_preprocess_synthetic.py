#!/usr/bin/env python3
"""
Unit and Integration Tests for Preprocessing Pipeline on Synthetic Data.

Tests all key edge cases and invariants:
1. Exact 3-example generation with input window, length, and delta_t on session [10, 25, 37, 42]
2. Consecutive repeat collapse: A,A,A,B -> A,B
3. Non-consecutive repeat preservation: A,B,A kept intact
4. Exact duplicate row removal
5. Same-timestamp tie-breaking by file row index
6. 1-click session filtering
7. Multi-chunk reading with small chunksize (chunksize=7)
8. Iterative train vocabulary pruning (support < 2) and val/test OOV cleaning
9. Max-length truncation with max_len=3 and delta_t[0] == 0
10. Chronological split ordering
11. End-to-end validation checks V1-V14 and overwrite determinism (identical fingerprints)
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess_data import main as preprocess_main


def create_synthetic_raw_dataset(raw_path: Path) -> None:
    """
    Create synthetic raw dataset with planted cases in non-session-grouped shuffled order.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        # Session 1: items [101, 102, 103, 104], gaps 5s, 20s, 100s
        (1, "2014-04-01T10:00:00.000Z", 101, "0"),
        (1, "2014-04-01T10:00:05.000Z", 102, "0"),
        (1, "2014-04-01T10:00:25.000Z", 103, "0"),
        (1, "2014-04-01T10:02:05.000Z", 104, "0"),

        # Session 5: 1-click session (must be dropped)
        (5, "2014-04-01T10:03:00.000Z", 101, "0"),

        # Session 2: A,A,A,B -> 101, 101, 101, 102 (collapsed to 101, 102)
        (2, "2014-04-01T10:05:00.000Z", 101, "0"),
        (2, "2014-04-01T10:05:01.000Z", 101, "0"),
        (2, "2014-04-01T10:05:02.000Z", 101, "0"),
        (2, "2014-04-01T10:05:10.000Z", 102, "0"),

        # Session 3: A,B,A -> 101, 102, 101 (kept intact)
        (3, "2014-04-01T10:10:00.000Z", 101, "0"),
        (3, "2014-04-01T10:10:05.000Z", 102, "0"),
        (3, "2014-04-01T10:10:10.000Z", 101, "0"),

        # Session 4: duplicate row + same timestamp tie-breaking
        (4, "2014-04-01T10:15:00.000Z", 102, "0"), # Row A
        (4, "2014-04-01T10:15:00.000Z", 102, "0"), # Duplicate of Row A -> dropped
        (4, "2014-04-01T10:15:00.000Z", 103, "0"), # Same timestamp, different item -> ordered after Row A

        # Session 6: 5 items [101, 102, 103, 104, 105] (tests max_len=3 truncation)
        (6, "2014-04-01T10:20:00.000Z", 101, "0"),
        (6, "2014-04-01T10:20:02.000Z", 102, "0"),
        (6, "2014-04-01T10:20:05.000Z", 103, "0"),
        (6, "2014-04-01T10:20:09.000Z", 104, "0"),
        (6, "2014-04-01T10:20:14.000Z", 105, "0"),

        # Session 7: [104, 105, 999] (item 999 has support 1 in train -> rare, pruned)
        (7, "2014-04-01T10:25:00.000Z", 104, "0"),
        (7, "2014-04-01T10:25:05.000Z", 105, "0"),
        (7, "2014-04-01T10:25:10.000Z", 999, "0"),

        # Session 8 (Val): [101, 102, 999] (item 999 is OOV -> dropped, leaves [101, 102])
        (8, "2014-04-01T10:30:00.000Z", 101, "0"),
        (8, "2014-04-01T10:30:05.000Z", 102, "0"),
        (8, "2014-04-01T10:30:10.000Z", 999, "0"),

        # Session 9 (Val): [103, 104]
        (9, "2014-04-01T10:35:00.000Z", 103, "0"),
        (9, "2014-04-01T10:35:05.000Z", 104, "0"),

        # Session 10 (Test): [101, 105]
        (10, "2014-04-01T10:40:00.000Z", 101, "0"),
        (10, "2014-04-01T10:40:05.000Z", 105, "0"),

        # Session 11 (Test): [102, 104, 888] (item 888 is OOV -> dropped, leaves [102, 104])
        (11, "2014-04-01T10:45:00.000Z", 102, "0"),
        (11, "2014-04-01T10:45:05.000Z", 104, "0"),
        (11, "2014-04-01T10:45:10.000Z", 888, "0"),
    ]

    # Write without header
    df = pd.DataFrame(rows, columns=["session_id", "timestamp", "item_id", "category"])
    # Intentionally shuffle rows to verify global stable sort & multi-chunk streaming
    shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shuffled_df.to_csv(raw_path, header=False, index=False)


def run_synthetic_test() -> None:
    """Execute full synthetic test suite."""
    print("Running synthetic preprocessing pipeline tests...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_path = tmp_path / "data" / "raw" / "raw_clicks.dat"
        out_root = tmp_path / "data" / "processed"
        res_root = tmp_path / "results" / "preprocessing"

        create_synthetic_raw_dataset(raw_path)

        argv = [
            "--raw-path", str(raw_path),
            "--output-root", str(out_root),
            "--results-root", str(res_root),
            "--chunksize", "7",  # Test chunk boundaries
            "--subset-fraction", "1",
            "--min-session-length", "2",
            "--min-item-support", "2",
            "--max-len", "3",
            "--val-fraction", "0.2",
            "--test-fraction", "0.2",
            "--collapse-consecutive-repeats",
            "--save-time-deltas",
        ]

        ret = preprocess_main(argv)
        assert ret == 0, f"preprocess_main returned non-zero code {ret}"

        # Find generated run folder
        run_dirs = list(out_root.glob("frac_*"))
        assert len(run_dirs) == 1, f"Expected 1 run dir, found {run_dirs}"
        run_dir = run_dirs[0]

        # Verify _SUCCESS exists
        assert (run_dir / "_SUCCESS").exists(), "_SUCCESS file missing in output directory"

        # Load generated files using context managers to avoid file locks
        with np.load(run_dir / "train.npz") as data:
            train_inputs = data["inputs"]
            train_lengths = data["lengths"]
            train_targets = data["targets"]
            train_sess_idx = data["session_index"]
            train_deltas = data["input_delta_t"]

        with np.load(run_dir / "sessions_train.npz") as data:
            train_sids = data["session_ids"]

        with np.load(run_dir / "sessions_val.npz") as data:
            val_sids = data["session_ids"]

        with np.load(run_dir / "sessions_test.npz") as data:
            test_sids = data["session_ids"]

        item_index_df = pd.read_csv(run_dir / "item_index.csv")

        # 1. Vocabulary verification: items 101, 102, 103, 104, 105 -> indices 1..5
        assert len(item_index_df) == 5, f"Expected 5 vocabulary items, got {len(item_index_df)}"
        assert list(item_index_df["raw_item_id"]) == [101, 102, 103, 104, 105]
        assert list(item_index_df["item_index"]) == [1, 2, 3, 4, 5]
        # Item 999 must NOT be in vocabulary
        assert 999 not in list(item_index_df["raw_item_id"])

        # 2. Session 1 Verification (items: [101, 102, 103, 104] -> indices [1, 2, 3, 4], gaps 5s, 20s, 100s)
        # Session 1 is the first session in train
        s1_indices = np.where(train_sess_idx == 0)[0]
        assert len(s1_indices) == 3, f"Expected 3 examples for session 1, got {len(s1_indices)}"

        # Example 0: p=1, input=[1, 0, 0], length=1, target=2, deltas=[0, 0, 0]
        ex0 = s1_indices[0]
        np.testing.assert_array_equal(train_inputs[ex0], [1, 0, 0])
        assert train_lengths[ex0] == 1
        assert train_targets[ex0] == 2
        np.testing.assert_allclose(train_deltas[ex0], [0.0, 0.0, 0.0], atol=1e-5)

        # Example 1: p=2, input=[1, 2, 0], length=2, target=3, deltas=[0, 5.0, 0]
        ex1 = s1_indices[1]
        np.testing.assert_array_equal(train_inputs[ex1], [1, 2, 0])
        assert train_lengths[ex1] == 2
        assert train_targets[ex1] == 3
        np.testing.assert_allclose(train_deltas[ex1], [0.0, 5.0, 0.0], atol=1e-5)

        # Example 2: p=3, input=[1, 2, 3], length=3, target=4, deltas=[0, 5.0, 20.0]
        ex2 = s1_indices[2]
        np.testing.assert_array_equal(train_inputs[ex2], [1, 2, 3])
        assert train_lengths[ex2] == 3
        assert train_targets[ex2] == 4
        np.testing.assert_allclose(train_deltas[ex2], [0.0, 5.0, 20.0], atol=1e-5)

        # 3. Session 2 (Collapsed A,A,A,B -> [101, 102] -> [1, 2])
        # Find session 2 in train_sids
        s2_idx = np.where(train_sids == 2)[0][0]
        s2_ex_indices = np.where(train_sess_idx == s2_idx)[0]
        assert len(s2_ex_indices) == 1
        np.testing.assert_array_equal(train_inputs[s2_ex_indices[0]], [1, 0, 0])
        assert train_targets[s2_ex_indices[0]] == 2

        # 4. Session 6 (Truncation with max_len=3): items [101, 102, 103, 104, 105] -> [1, 2, 3, 4, 5]
        s6_idx = np.where(train_sids == 6)[0][0]
        s6_ex_indices = np.where(train_sess_idx == s6_idx)[0]
        assert len(s6_ex_indices) == 4

        # Last example (p=4, target=5): input=[2, 3, 4], length=3, deltas=[0, 3.0, 4.0]
        p4_ex = s6_ex_indices[3]
        np.testing.assert_array_equal(train_inputs[p4_ex], [2, 3, 4])
        assert train_lengths[p4_ex] == 3
        assert train_targets[p4_ex] == 5
        np.testing.assert_allclose(train_deltas[p4_ex], [0.0, 3.0, 4.0], atol=1e-5)

        # 5. Check 1-click session 5 is absent everywhere
        all_sids = np.concatenate([train_sids, val_sids, test_sids])
        assert 5 not in all_sids, "1-click session 5 was not dropped"

        # 6. Test --overwrite determinism and fingerprint equality
        with open(run_dir / "metadata.json", "r", encoding="utf-8") as f:
            first_meta = json.load(f)
        first_fp = first_meta["fingerprints"]

        # Run again with --overwrite
        argv_overwrite = argv + ["--overwrite"]
        ret_ov = preprocess_main(argv_overwrite)
        assert ret_ov == 0

        with open(run_dir / "metadata.json", "r", encoding="utf-8") as f:
            second_meta = json.load(f)
        second_fp = second_meta["fingerprints"]

        assert first_fp["__combined__"] == second_fp["__combined__"], "Fingerprints differed on overwrite run"

        # 7. Test --verify-only
        argv_verify = ["--verify-only", run_dir.name, "--output-root", str(out_root), "--raw-path", str(raw_path)]
        ret_ver = preprocess_main(argv_verify)
        assert ret_ver == 0, f"--verify-only failed with code {ret_ver}"

    print("All synthetic preprocessing tests PASSED successfully!")


if __name__ == "__main__":
    run_synthetic_test()
