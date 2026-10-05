#!/usr/bin/env python3
"""
Unit tests for ranking metrics (Recall@K, MRR@K).

Tests exact hand-computed rank values, pessimistic tie handling, PAD exclusion,
and NaN exception handling.
"""

from pathlib import Path
import sys
import numpy as np
import torch

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.metrics import compute_ranking_metrics


def test_metrics_exact() -> None:
    """Test hand-computed score matrix with targets at specific ranks."""
    # 6 examples, vocab size = 15 (PAD = index 0)
    # Vocab items: 0 (PAD), 1..14
    batch_size = 6
    vocab_size = 15
    logits = np.zeros((batch_size, vocab_size), dtype=np.float32)

    # Ex 0: target at rank 1. Target item = 1. Scores: item 1 has score 10.0, others 0.0
    logits[0, 1] = 10.0
    target_0 = 1

    # Ex 1: target at rank 2. Target item = 2. Scores: item 1 has score 10.0, item 2 has 9.0, others 0.0
    logits[1, 1] = 10.0
    logits[1, 2] = 9.0
    target_1 = 2

    # Ex 2: target at rank 5. Target item = 5. Items 1,2,3,4 have score 10.0, item 5 has 5.0
    for item in [1, 2, 3, 4]:
        logits[2, item] = 10.0
    logits[2, 5] = 5.0
    target_2 = 5

    # Ex 3: target at rank 6. Target item = 6. Items 1,2,3,4,5 have score 10.0, item 6 has 5.0
    for item in [1, 2, 3, 4, 5]:
        logits[3, item] = 10.0
    logits[3, 6] = 5.0
    target_3 = 6

    # Ex 4: target at rank 10. Target item = 10. Items 1..9 have score 10.0, item 10 has 5.0
    for item in range(1, 10):
        logits[4, item] = 10.0
    logits[4, 10] = 5.0
    target_4 = 10

    # Ex 5: target at rank 11. Target item = 11. Items 1..10 have score 10.0, item 11 has 5.0
    for item in range(1, 11):
        logits[5, item] = 10.0
    logits[5, 11] = 5.0
    target_5 = 11

    targets = np.array([target_0, target_1, target_2, target_3, target_4, target_5])

    res = compute_ranking_metrics(logits, targets, k_values=(5, 10), pad_index=0)

    # Hand-computed ranks for examples: [1, 2, 5, 6, 10, 11]
    # Recall@5: ranks <= 5 are [1, 2, 5] -> 3 / 6 = 0.5
    # MRR@5: (1/1 + 1/2 + 1/5) / 6 = (1.0 + 0.5 + 0.2) / 6 = 1.7 / 6 = 0.283333...
    # Recall@10: ranks <= 10 are [1, 2, 5, 6, 10] -> 5 / 6 = 0.833333...
    # MRR@10: (1/1 + 1/2 + 1/5 + 1/6 + 1/10) / 6 = (1 + 0.5 + 0.2 + 0.166667 + 0.1) / 6 = 1.966667 / 6 = 0.327777...

    np.testing.assert_allclose(res["Recall@5"], 3.0 / 6.0, atol=1e-5)
    np.testing.assert_allclose(res["MRR@5"], 1.7 / 6.0, atol=1e-5)
    np.testing.assert_allclose(res["Recall@10"], 5.0 / 6.0, atol=1e-5)
    np.testing.assert_allclose(res["MRR@10"], 1.9666666666666668 / 6.0, atol=1e-5)
    print("test_metrics_exact PASSED")


def test_pessimistic_ties() -> None:
    """Test that constant scores give pessimistic rank equal to total non-pad items."""
    batch_size = 4
    vocab_size = 15  # 14 real items
    logits = np.zeros((batch_size, vocab_size), dtype=np.float32)  # All constant
    targets = np.array([1, 2, 3, 4])

    res = compute_ranking_metrics(logits, targets, k_values=(5, 10), pad_index=0)

    # Constant scores mean all 14 non-PAD items have score >= target_score.
    # So rank for each item is 14.
    # Since 14 > 10, Recall@5 = 0, Recall@10 = 0, MRR@5 = 0, MRR@10 = 0.
    assert res["Recall@5"] == 0.0
    assert res["Recall@10"] == 0.0
    assert res["MRR@5"] == 0.0
    assert res["MRR@10"] == 0.0
    print("test_pessimistic_ties PASSED")


def test_pad_exclusion() -> None:
    """Test that PAD column is never counted above target even if its logit is huge."""
    batch_size = 1
    vocab_size = 5
    logits = np.zeros((batch_size, vocab_size), dtype=np.float32)
    logits[0, 0] = 9999.0  # Huge logit on PAD
    logits[0, 1] = 5.0     # Target item 1 has logit 5.0
    targets = np.array([1])

    res = compute_ranking_metrics(logits, targets, k_values=(5, 10), pad_index=0)

    # Item 1 should have rank 1 because PAD is excluded
    assert res["Recall@5"] == 1.0
    assert res["MRR@5"] == 1.0
    print("test_pad_exclusion PASSED")


def test_nan_handling() -> None:
    """Test that NaN in logits raises ValueError."""
    logits = np.zeros((2, 5), dtype=np.float32)
    logits[0, 2] = np.nan
    targets = np.array([1, 2])

    try:
        compute_ranking_metrics(logits, targets)
        assert False, "Should have raised ValueError on NaN logits"
    except ValueError:
        print("test_nan_handling PASSED")


if __name__ == "__main__":
    test_metrics_exact()
    test_pessimistic_ties()
    test_pad_exclusion()
    test_nan_handling()
    print("All test_metrics.py tests PASSED successfully!")
