"""Integration and invariant unit tests for model architecture, data loading, and training pipeline."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess_data import main as preprocess_main
from scripts.train import main as train_main
from src.models import build_model
from src.models.base import SequenceEncoder, SessionRecommender
from src.models.gru import GRUEncoder
from src.utils.data import load_run, BatchIterator


class DummyEncoder(SequenceEncoder):
    """Dummy sequence encoder to verify pluggable interface (T6)."""

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)

    def forward(
        self,
        emb: torch.Tensor,
        lengths: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Simple mean over sequence for testing
        return self.proj(emb.mean(dim=1))


def create_synthetic_raw_cycle_dataset(raw_path: Path, num_sessions: int = 400) -> None:
    """Create a synthetic raw CSV dataset with deterministic cycle: next = (prev % 30) + 1."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    base_timestamp = 1400000000  # epoch seconds

    rng = np.random.default_rng(42)

    for session_id in range(1, num_sessions + 1):
        seq_len = rng.integers(4, 9)
        start_item = rng.integers(1, 31)

        curr_item = start_item
        for t in range(seq_len):
            ts = base_timestamp + session_id * 100 + t * 5
            # ISO timestamp string
            dt_str = pd_timestamp_str(ts)
            rows.append((session_id, dt_str, int(curr_item), "0"))
            curr_item = (curr_item % 30) + 1

    with open(raw_path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")


def pd_timestamp_str(ts: int) -> str:
    import datetime

    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def test_t1_padding_invariance(run_dir: Path) -> None:
    """T1: Check logits for an example are unchanged when batched/padded differently."""
    print("Running T1: Padding invariance check...")
    data = load_run(run_dir, splits=("train",), verify=True)
    vocab_size = data["vocab_size"]

    model = build_model(
        name="gru",
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.0,
        use_time_deltas=False,
    )
    model.eval()

    # Single short example (len=3)
    inputs1 = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
    lengths1 = torch.tensor([3], dtype=torch.long)

    # Batched example with a longer sequence (len=5)
    inputs2 = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.long)
    lengths2 = torch.tensor([3, 5], dtype=torch.long)

    # Batched example where padding region has non-zero items
    inputs3 = torch.tensor([[1, 2, 3, 9, 9]], dtype=torch.long)
    lengths3 = torch.tensor([3], dtype=torch.long)

    with torch.no_grad():
        out1 = model({"inputs": inputs1, "lengths": lengths1})[0]
        out2 = model({"inputs": inputs2, "lengths": lengths2})[0]
        out3 = model({"inputs": inputs3, "lengths": lengths3})[0]

    assert torch.allclose(out1, out2, atol=1e-6), "Logits changed when batched with longer sequence!"
    assert torch.allclose(out1, out3, atol=1e-6), "Logits changed when padding region filled with non-zero items!"
    print("  T1 PASSED!")


def test_t2_item_only_isolation(run_dir: Path) -> None:
    """T2: Item-only mode must be completely bit-identical regardless of delta_t."""
    print("Running T2: Item-only isolation check...")
    data = load_run(run_dir, splits=("train",), verify=True)
    vocab_size = data["vocab_size"]

    model = build_model(
        name="gru",
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.0,
        use_time_deltas=False,
    )
    model.eval()

    inputs = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([3], dtype=torch.long)
    noise_delta = torch.randn(1, 5, dtype=torch.float32)

    with torch.no_grad():
        out_none = model({"inputs": inputs, "lengths": lengths, "delta_t": None})
        out_noise = model({"inputs": inputs, "lengths": lengths, "delta_t": noise_delta})

    assert torch.equal(out_none, out_noise), "Item-only mode affected by delta_t input!"
    print("  T2 PASSED!")


def test_t3_time_plumbing(run_dir: Path) -> None:
    """T3: Time-aware model must change output when delta_t is modified."""
    print("Running T3: Time plumbing check...")
    data = load_run(run_dir, splits=("train",), verify=True)
    vocab_size = data["vocab_size"]

    model = build_model(
        name="gru",
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dim=64,
        dropout=0.0,
        use_time_deltas=True,
    )
    model.eval()

    inputs = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([3], dtype=torch.long)
    delta1 = torch.zeros((1, 5), dtype=torch.float32)
    delta2 = torch.ones((1, 5), dtype=torch.float32) * 5.0

    with torch.no_grad():
        out1 = model({"inputs": inputs, "lengths": lengths, "delta_t": delta1})
        out2 = model({"inputs": inputs, "lengths": lengths, "delta_t": delta2})

    # Mask column 0 is -inf, so ignore it for numerical comparison (-inf - -inf = nan)
    diff = torch.max(torch.abs(out1[:, 1:] - out2[:, 1:])).item()
    assert diff > 1e-4, f"Time-aware model logits did not change enough when delta_t changed (diff: {diff})"
    print("  T3 PASSED!")


def test_t4_learning_sanity(run_dir: Path, temp_dir: Path) -> None:
    """T4: GRU trained a few epochs on CPU reaches val Recall@5 > 0.8 on cycle data."""
    print("Running T4: Learning sanity check...")
    out_exp_dir = temp_dir / "results"
    ckpt_dir = temp_dir / "checkpoints"

    run_name = run_dir.name
    data_root = run_dir.parent

    argv = [
        "--run-name", run_name,
        "--data-root", str(data_root),
        "--model", "gru",
        "--embed-dim", "32",
        "--hidden-dim", "64",
        "--dropout", "0.0",
        "--lr", "1e-2",
        "--batch-size", "32",
        "--epochs", "15",
        "--patience", "15",
        "--device", "cpu",
        "--output-root", str(out_exp_dir),
        "--checkpoint-root", str(ckpt_dir),
        "--overwrite",
    ]

    ret = train_main(argv)
    assert ret == 0, "train.py execution failed!"

    metrics_file = out_exp_dir / run_name / "gru_item" / "seed0" / "metrics_val.json"
    assert metrics_file.exists(), f"metrics_val.json missing at {metrics_file}"

    with open(metrics_file, "r", encoding="utf-8") as f:
        val_res = json.load(f)

    rec5 = val_res["metrics"]["Recall@5"]
    print(f"  T4 val Recall@5: {rec5:.4f}")
    assert rec5 > 0.80, f"Learning sanity check failed: val Recall@5 ({rec5:.4f}) <= 0.80"
    print("  T4 PASSED!")


def test_t5_data_verification(run_dir: Path, temp_dir: Path) -> None:
    """T5: Reject folders missing _SUCCESS or with corrupted arrays."""
    print("Running T5: Data verification check...")

    # Case A: Missing _SUCCESS
    no_success_dir = temp_dir / "no_success_run"
    shutil.copytree(run_dir, no_success_dir)
    (no_success_dir / "_SUCCESS").unlink()

    try:
        load_run(no_success_dir, verify=True)
        assert False, "Failed to raise error on missing _SUCCESS!"
    except (ValueError, FileNotFoundError) as e:
        assert "_SUCCESS" in str(e)

    # Case B: Corrupted array fingerprint
    corrupt_dir = temp_dir / "corrupt_run"
    shutil.copytree(run_dir, corrupt_dir)

    train_npz = corrupt_dir / "train.npz"
    loaded = dict(np.load(train_npz))
    loaded["inputs"] = loaded["inputs"].copy()
    loaded["inputs"][0, 0] = 99999  # mutate array
    np.savez_compressed(train_npz, **loaded)

    try:
        load_run(corrupt_dir, verify=True)
        assert False, "Failed to raise error on corrupted array!"
    except ValueError as e:
        assert "mismatch" in str(e).lower()

    print("  T5 PASSED!")


def test_t6_encoder_interface() -> None:
    """T6: SessionRecommender works with custom dummy encoder interface."""
    print("Running T6: Pluggable encoder interface check...")
    dummy_enc = DummyEncoder(embed_dim=32, hidden_dim=64)
    model = SessionRecommender(
        vocab_size=100,
        embed_dim=32,
        encoder=dummy_enc,
        hidden_dim=64,
    )
    model.eval()

    batch = {
        "inputs": torch.randint(1, 100, (4, 10)),
        "lengths": torch.tensor([5, 8, 10, 3]),
    }
    logits = model(batch)
    assert logits.shape == (4, 100), f"Unexpected logits shape: {logits.shape}"
    print("  T6 PASSED!")


def main() -> None:
    print("Starting Model Pipeline Test Suite...")
    with tempfile.TemporaryDirectory() as tmp_str:
        temp_dir = Path(tmp_str)
        raw_path = temp_dir / "data" / "raw" / "yoochoose_clicks.dat"
        proc_dir = temp_dir / "data" / "processed"
        res_dir = temp_dir / "results" / "preprocessing"

        print("Creating synthetic dataset...")
        create_synthetic_raw_cycle_dataset(raw_path, num_sessions=400)

        print("Preprocessing synthetic dataset...")
        preprocess_argv = [
            "--raw-path", str(raw_path),
            "--output-root", str(proc_dir),
            "--results-root", str(res_dir),
            "--subset-fraction", "1",
            "--min-item-support", "2",
            "--max-len", "5",
            "--val-fraction", "0.2",
            "--test-fraction", "0.2",
            "--overwrite",
        ]
        ret = preprocess_main(preprocess_argv)
        assert ret == 0, "Preprocessing synthetic run failed!"

        run_dirs = list(proc_dir.glob("frac_*"))
        assert len(run_dirs) == 1, f"Expected 1 run directory, found {run_dirs}"
        run_dir = run_dirs[0]

        test_t1_padding_invariance(run_dir)
        test_t2_item_only_isolation(run_dir)
        test_t3_time_plumbing(run_dir)
        test_t4_learning_sanity(run_dir, temp_dir)
        test_t5_data_verification(run_dir, temp_dir)
        test_t6_encoder_interface()

    print("\nALL MODEL PIPELINE TESTS (T1-T6) PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
