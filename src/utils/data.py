"""
Data loading, verification, batch iteration, and delta-t normalization utilities.

Provides functions to load processed runs with metadata verification, batch iterators
for training and evaluation, and normalized time-delta features.
"""

from pathlib import Path
import json
import logging
from typing import Any, Dict, Iterator, Optional, Sequence, Union
import numpy as np
import torch

from src.utils.preprocessing import (
    compute_array_fingerprint,
    compute_directory_fingerprints,
)

logger = logging.getLogger(__name__)


def load_run(
    run_dir: Union[str, Path],
    splits: Sequence[str] = ("train", "val"),
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Load a processed dataset run directory with optional fingerprint verification.

    Args:
        run_dir: Path to processed run directory (e.g. data/processed/<run_name>).
        splits: Sequence of split names to load (e.g. ('train', 'val')).
        verify: If True, recomputes content fingerprints and compares with metadata.json.

    Returns:
        Structured dictionary containing metadata, extracted metrics, and loaded split data.
    """
    path = Path(run_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {path}")

    success_file = path / "_SUCCESS"
    meta_file = path / "metadata.json"

    if not success_file.exists():
        raise ValueError(f"Run directory is incomplete or unvalidated (missing _SUCCESS marker): {path}")

    if not meta_file.exists():
        raise FileNotFoundError(f"metadata.json missing in run directory: {path}")

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not metadata.get("validation_passed", False):
        raise ValueError(f"metadata.json indicates run validation failed: {path}")

    # Extract required metadata fields with strict checking
    try:
        vocab_size = int(metadata["vocabulary"]["vocab_size"])
        pad_index = int(metadata["vocabulary"]["pad_index"])
        max_len = int(metadata["resolved_args"]["max_len"])
        num_items = int(metadata["vocabulary"]["num_items"])
        combined_fingerprint = metadata["fingerprints"].get("__combined__")
        oov_stats = metadata.get("oov", {})
        time_delta_stats = metadata.get("time_delta_stats", {})
    except KeyError as e:
        raise KeyError(f"Missing expected metadata key in {meta_file}: {e}")

    # Fingerprint verification if requested
    if verify:
        logger.info("Verifying data fingerprints for %s...", path.name)
        on_disk_fps = compute_directory_fingerprints(path)
        meta_fps = metadata.get("fingerprints", {})

        for k, expected_fp in meta_fps.items():
            if k not in on_disk_fps:
                raise ValueError(f"Fingerprint check failed: file/key '{k}' missing on disk in {path}")
            if on_disk_fps[k] != expected_fp:
                raise ValueError(
                    f"Fingerprint mismatch for '{k}' in {path}:\n"
                    f"  Expected: {expected_fp}\n"
                    f"  Computed: {on_disk_fps[k]}"
                )
        logger.info("Fingerprint verification PASSED for %s", path.name)

    # Load requested splits
    splits_data: Dict[str, Dict[str, np.ndarray]] = {}
    sessions_data: Dict[str, Dict[str, np.ndarray]] = {}

    for split in splits:
        ex_file = path / f"{split}.npz"
        sess_file = path / f"sessions_{split}.npz"

        if not ex_file.exists():
            raise FileNotFoundError(f"Split file missing: {ex_file}")
        if not sess_file.exists():
            raise FileNotFoundError(f"Sessions file missing: {sess_file}")

        with np.load(ex_file) as d_ex:
            splits_data[split] = {key: d_ex[key] for key in d_ex.files}

        with np.load(sess_file) as d_sess:
            sessions_data[split] = {key: d_sess[key] for key in d_sess.files}

    return {
        "run_dir": path,
        "run_name": metadata.get("run_name", path.name),
        "vocab_size": vocab_size,
        "pad_index": pad_index,
        "max_len": max_len,
        "num_items": num_items,
        "combined_fingerprint": combined_fingerprint,
        "oov_stats": oov_stats,
        "time_delta_stats": time_delta_stats,
        "metadata": metadata,
        "splits": splits_data,
        "sessions": sessions_data,
    }


def normalize_delta_t(
    delta_t: Union[np.ndarray, torch.Tensor],
    lengths: Union[np.ndarray, torch.Tensor],
    stats: Dict[str, Any],
) -> torch.Tensor:
    """
    Normalize time-delta features using log1p scaling by train std_log1p (no mean centering).
    Padded positions (j >= length) are forced to 0.0.

    Args:
        delta_t: Input time gaps in seconds, shape [B, T].
        lengths: Active sequence lengths, shape [B].
        stats: Dictionary containing 'log1p_std' (or 'std_log1p').

    Returns:
        Normalized time-delta tensor of shape [B, T], dtype float32.
    """
    if not isinstance(delta_t, torch.Tensor):
        dt_tensor = torch.from_numpy(np.asarray(delta_t, dtype=np.float32))
    else:
        dt_tensor = delta_t.to(torch.float32)

    if not isinstance(lengths, torch.Tensor):
        len_tensor = torch.from_numpy(np.asarray(lengths, dtype=np.int64))
    else:
        len_tensor = lengths.to(torch.int64)

    std_val = float(stats.get("log1p_std", stats.get("std_log1p", 1.0)))
    if std_val <= 0.0 or np.isnan(std_val):
        std_val = 1.0

    # Scale: log1p(delta_t) / std (NO mean centering)
    log1p_dt = torch.log1p(dt_tensor.clamp(min=0.0))
    normalized = log1p_dt / std_val

    # Mask padded positions: positions >= length are forced to exactly 0.0
    B, T = dt_tensor.shape
    seq_indices = torch.arange(T, device=dt_tensor.device).unsqueeze(0)  # [1, T]
    mask = seq_indices < len_tensor.unsqueeze(1)  # [B, T]
    normalized = torch.where(mask, normalized, torch.zeros_like(normalized))

    return normalized


class BatchIterator:
    """
    Batch iterator for training and evaluation without PyTorch DataLoader workers.
    """

    def __init__(
        self,
        split_data: Dict[str, np.ndarray],
        batch_size: int,
        is_train: bool = True,
        shuffle: Optional[bool] = None,
        seed: int = 0,
        epoch: int = 0,
        use_time_deltas: bool = False,
        time_delta_stats: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        self.inputs = split_data["inputs"]
        self.lengths = split_data["lengths"]
        self.targets = split_data["targets"]
        self.use_time_deltas = use_time_deltas
        self.time_delta_stats = time_delta_stats
        self.delta_t = split_data.get("input_delta_t") if use_time_deltas else None
        self.batch_size = batch_size
        self.is_train = shuffle if shuffle is not None else is_train
        self.seed = seed
        self.epoch = epoch
        self.device = device or torch.device("cpu")

        self.n_samples = len(self.targets)

        if self.is_train:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.indices = torch.randperm(self.n_samples, generator=g).numpy()
        else:
            self.indices = np.arange(self.n_samples, dtype=np.int64)

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size if self.batch_size > 0 else 0

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = self.indices[i : i + self.batch_size]

            b_inputs = self.inputs[batch_idx]
            b_lengths = self.lengths[batch_idx]
            b_targets = self.targets[batch_idx]

            # Dynamic batch trimming to max length in this batch
            T = int(np.max(b_lengths)) if len(b_lengths) > 0 else 0

            inp_tensor = torch.tensor(b_inputs[:, :T], dtype=torch.long, device=self.device)
            len_tensor = torch.tensor(b_lengths, dtype=torch.long, device=self.device)
            tgt_tensor = torch.tensor(b_targets, dtype=torch.long, device=self.device)

            dt_tensor = None
            if self.use_time_deltas and self.delta_t is not None:
                b_dt = self.delta_t[batch_idx, :T]
                dt_raw = torch.tensor(b_dt, dtype=torch.float32, device=self.device)
                if self.time_delta_stats is not None:
                    dt_tensor = normalize_delta_t(dt_raw, len_tensor, self.time_delta_stats)
                else:
                    dt_tensor = dt_raw

            yield {
                "inputs": inp_tensor,
                "lengths": len_tensor,
                "targets": tgt_tensor,
                "delta_t": dt_tensor,
            }
