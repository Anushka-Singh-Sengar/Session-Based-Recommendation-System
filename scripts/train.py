"""Training script for session-based recommender models."""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import build_model
from src.utils.data import load_run, BatchIterator
from src.utils.metrics import compute_ranking_metrics

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure standard logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def seed_everything(seed: int) -> None:
    """Set random seed across python, numpy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_info() -> Dict[str, Any]:
    """Retrieve git commit hash and dirty status if inside a git repository."""
    import subprocess

    git_info = {"commit": "unknown", "is_dirty": False}
    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        git_info["commit"] = commit_res.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        git_info["is_dirty"] = len(status_res.stdout.strip()) > 0
    except Exception:
        pass
    return git_info


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train session-based recommendation models."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name of processed run directory in data-root.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/processed",
        help="Path to data root directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gru",
        help="Sequence encoder architecture (e.g. 'gru').",
    )
    parser.add_argument(
        "--use-time-deltas",
        action="store_true",
        dest="use_time_deltas",
        default=False,
        help="Include normalized input time deltas.",
    )
    parser.add_argument(
        "--no-use-time-deltas",
        action="store_false",
        dest="use_time_deltas",
        help="Item-only mode without time deltas.",
    )
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Train batch size.")
    parser.add_argument(
        "--eval-batch-size", type=int, default=1024, help="Validation batch size."
    )
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs.")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience epochs."
    )
    parser.add_argument(
        "--clip-norm", type=float, default=5.0, help="Gradient clipping max norm."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Experiment tag name. Defaults to '{model}_{item|itemdt}'.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/experiments",
        help="Root output directory for metrics/configs.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="checkpoints",
        help="Root checkpoint directory for model weights.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if present.",
    )
    parser.add_argument(
        "--no-verify-data",
        action="store_true",
        help="Skip data fingerprint verification.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Maximum training batches per epoch (for smoke testing only).",
    )
    return parser.parse_args(argv)


def evaluate_split(
    model: nn.Module,
    split_data: Dict[str, np.ndarray],
    batch_size: int,
    use_time_deltas: bool,
    time_delta_stats: Dict[str, float],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a given dataset split."""
    model.eval()
    val_iterator = BatchIterator(
        split_data,
        batch_size=batch_size,
        shuffle=False,
        use_time_deltas=use_time_deltas,
        time_delta_stats=time_delta_stats,
        device=device,
    )

    total_examples = 0
    sum_metrics: Dict[str, float] = {}

    with torch.no_grad():
        for batch in val_iterator:
            logits = model(batch)
            b_size = batch["targets"].size(0)
            b_metrics = compute_ranking_metrics(
                logits, batch["targets"], k_values=(5, 10), pad_index=0
            )
            for k, v in b_metrics.items():
                sum_metrics[k] = sum_metrics.get(k, 0.0) + v * b_size
            total_examples += b_size

    if total_examples == 0:
        return {"Recall@5": 0.0, "Recall@10": 0.0, "MRR@5": 0.0, "MRR@10": 0.0}

    return {k: v / total_examples for k, v in sum_metrics.items()}


def main(argv: Optional[List[str]] = None) -> int:
    """Main training execution function.

    Returns:
        Exit code 0 on success, non-zero on failure.
    """
    setup_logging()
    args = parse_args(argv)

    # Resolve paths relative to REPO_ROOT if relative
    data_root = REPO_ROOT / args.data_root if not Path(args.data_root).is_absolute() else Path(args.data_root)
    output_root = REPO_ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    checkpoint_root = REPO_ROOT / args.checkpoint_root if not Path(args.checkpoint_root).is_absolute() else Path(args.checkpoint_root)

    run_dir = data_root / args.run_name
    info_mode = "item_plus_dt" if args.use_time_deltas else "item_only"
    tag = args.tag if args.tag is not None else f"{args.model}_{'itemdt' if args.use_time_deltas else 'item'}"

    output_dir = output_root / args.run_name / tag / f"seed{args.seed}"
    checkpoint_dir = checkpoint_root / args.run_name / tag / f"seed{args.seed}"

    if output_dir.exists() and not args.overwrite:
        logger.error(
            "Output directory already exists: %s. Use --overwrite to overwrite.",
            output_dir,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.max_train_batches is not None:
        logger.warning(
            "*** WARNING: --max-train-batches is set to %d. THIS IS A SMOKE TEST RUN! ***",
            args.max_train_batches,
        )

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Load dataset
    logger.info("Loading dataset run from: %s", run_dir)
    data = load_run(
        run_dir, splits=("train", "val"), verify=not args.no_verify_data
    )

    vocab_size = data["vocab_size"]
    pad_index = data["pad_index"]
    time_delta_stats = data["time_delta_stats"]
    combined_fingerprint = data["combined_fingerprint"]

    # Seed execution
    seed_everything(args.seed)

    # Build model
    model = build_model(
        name=args.model,
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_time_deltas=args.use_time_deltas,
        pad_idx=pad_index,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Built model '%s' with %d trainable parameters.", args.model, param_count)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mrr10 = -1.0
    best_epoch = 0
    patience_counter = 0

    history_rows = []
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()

        train_iterator = BatchIterator(
            data["splits"]["train"],
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=epoch,
            is_train=True,
            use_time_deltas=args.use_time_deltas,
            time_delta_stats=time_delta_stats,
            device=device,
        )

        running_loss = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(train_iterator, start=1):
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch["targets"])
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

        avg_train_loss = running_loss / max(total_batches, 1)

        # Validation evaluation
        val_metrics = evaluate_split(
            model=model,
            split_data=data["splits"]["val"],
            batch_size=args.eval_batch_size,
            use_time_deltas=args.use_time_deltas,
            time_delta_stats=time_delta_stats,
            device=device,
        )

        epoch_seconds = time.time() - epoch_start_time
        val_mrr10 = val_metrics["MRR@10"]

        logger.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val Recall@5: %.4f | Val Recall@10: %.4f | Val MRR@5: %.4f | Val MRR@10: %.4f | Time: %.2fs",
            epoch,
            args.epochs,
            avg_train_loss,
            val_metrics["Recall@5"],
            val_metrics["Recall@10"],
            val_metrics["MRR@5"],
            val_mrr10,
            epoch_seconds,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_Recall@5": val_metrics["Recall@5"],
                "val_Recall@10": val_metrics["Recall@10"],
                "val_MRR@5": val_metrics["MRR@5"],
                "val_MRR@10": val_mrr10,
                "seconds": round(epoch_seconds, 4),
            }
        )

        # Early stopping check based on Val MRR@10
        if val_mrr10 > best_val_mrr10:
            best_val_mrr10 = val_mrr10
            best_epoch = epoch
            patience_counter = 0
            best_ckpt_path = checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info("Saved new best checkpoint to %s (Val MRR@10: %.4f)", best_ckpt_path, val_mrr10)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", args.patience)
                break

    total_seconds = time.time() - total_start_time
    logger.info("Training complete in %.2fs. Best Epoch: %d (Val MRR@10: %.4f)", total_seconds, best_epoch, best_val_mrr10)

    # Restore best model for final validation output writing
    best_ckpt_path = checkpoint_dir / "best.pt"
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    final_val_metrics = evaluate_split(
        model=model,
        split_data=data["splits"]["val"],
        batch_size=args.eval_batch_size,
        use_time_deltas=args.use_time_deltas,
        time_delta_stats=time_delta_stats,
        device=device,
    )

    # Write history.csv
    history_file = output_dir / "history.csv"
    with open(history_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("epoch,train_loss,val_Recall@5,val_Recall@10,val_MRR@5,val_MRR@10,seconds\n")
        for row in history_rows:
            f.write(
                f"{row['epoch']},{row['train_loss']:.6f},{row['val_Recall@5']:.6f},"
                f"{row['val_Recall@10']:.6f},{row['val_MRR@5']:.6f},{row['val_MRR@10']:.6f},"
                f"{row['seconds']:.4f}\n"
            )

    # Write metrics_val.json
    val_metrics_out = {
        "best_epoch": best_epoch,
        "metrics": final_val_metrics,
    }
    with open(output_dir / "metrics_val.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(val_metrics_out, f, indent=2)

    # Write config.json
    val_oov_info = data.get("oov_stats", {}).get("val", {})
    val_oov_rate = val_oov_info.get("unseen_target_ratio", 0.0) if isinstance(val_oov_info, dict) else 0.0

    config_out = {
        "run_name": args.run_name,
        "tag": tag,
        "seed": args.seed,
        "info_mode": info_mode,
        "model": args.model,
        "use_time_deltas": args.use_time_deltas,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "clip_norm": args.clip_norm,
        "vocab_size": vocab_size,
        "combined_fingerprint": combined_fingerprint,
        "val_oov_rate": val_oov_rate,
        "parameter_count": param_count,
        "git": get_git_info(),
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": str(device),
        },
        "total_seconds": round(total_seconds, 4),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(config_out, f, indent=2)

    logger.info("Saved outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
