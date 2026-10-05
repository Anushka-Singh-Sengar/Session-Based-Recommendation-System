"""
Evaluation metrics for session-based recommendation.

Provides ranking metrics (Recall@K and MRR@K) computed over the full item vocabulary.
Ties in logits are handled pessimistically to ensure fair evaluation.
"""

from typing import Dict, Sequence, Union
import numpy as np
import torch


def compute_ranking_metrics(
    logits: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    k_values: Sequence[int] = (5, 10),
    pad_index: int = 0,
) -> Dict[str, float]:
    """
    Compute full-ranking Recall@K and MRR@K metrics for next-item prediction.

    Args:
        logits: Model output logits of shape [B, V] where V is vocab_size.
        targets: True target item indices of shape [B] with values in range [1, V-1].
        k_values: Sequence of cutoff ranks K (e.g. 5, 10).
        pad_index: Vocabulary index reserved for padding (default: 0).

    Returns:
        Dictionary mapping metric names (e.g. 'Recall@5', 'MRR@10') to float values.
    """
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = np.asarray(logits)

    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.asarray(targets)

    if np.isnan(logits_np).any() or np.isinf(logits_np).any():
        # Allow -inf only if it's explicitly placed on the pad_index column
        mask_non_pad = np.ones(logits_np.shape[1], dtype=bool)
        if 0 <= pad_index < logits_np.shape[1]:
            mask_non_pad[pad_index] = False
        non_pad_logits = logits_np[:, mask_non_pad]
        if np.isnan(non_pad_logits).any() or np.isinf(non_pad_logits).any():
            raise ValueError("Logits contain NaN or unexpected Infinite values.")

    batch_size, vocab_size = logits_np.shape
    if batch_size == 0:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ("Recall", "MRR")}

    # Mask out pad_index so it can never be ranked above target
    logits_clean = logits_np.copy()
    if 0 <= pad_index < vocab_size:
        logits_clean[:, pad_index] = -np.inf

    # Extract score of target for each item in batch
    row_indices = np.arange(batch_size)
    target_scores = logits_clean[row_indices, targets_np]  # [B]

    # Calculate pessimistic ranks: count items with score >= target_score
    # Target score itself satisfies >= target_score, so minimum rank is 1.
    # If multiple items share the same score, all are counted as >= target_score (pessimistic tie-breaking).
    ranks = np.sum(logits_clean >= target_scores[:, None], axis=1)  # [B]

    results: Dict[str, float] = {}
    for k in k_values:
        hits = (ranks <= k).astype(np.float64)
        recall_k = float(np.mean(hits))
        mrr_k = float(np.mean(hits / ranks))
        results[f"Recall@{k}"] = recall_k
        results[f"MRR@{k}"] = mrr_k

    return results
