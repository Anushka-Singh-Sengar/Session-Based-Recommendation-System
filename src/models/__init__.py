"""Models module initializer and model factory."""

from typing import Any

from src.models.base import SequenceEncoder, SessionRecommender
from src.models.gru import GRUEncoder


def build_model(
    name: str,
    vocab_size: int,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    use_time_deltas: bool = False,
    pad_idx: int = 0,
    **kwargs: Any,
) -> SessionRecommender:
    """Factory function to build a SessionRecommender with specified encoder architecture.

    Args:
        name: Name of the encoder ('gru').
        vocab_size: Total vocabulary size including PAD token.
        embed_dim: Embedding dimension for items.
        hidden_dim: Encoder hidden dimension.
        dropout: Dropout rate for embeddings and encoder output.
        use_time_deltas: Whether encoder uses time delta features.
        pad_idx: Padding index (default 0).

    Returns:
        SessionRecommender module instance.
    """
    model_name = name.lower()

    if model_name == "gru":
        encoder = GRUEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            use_time_deltas=use_time_deltas,
        )
    elif model_name in ("ltc", "cfc"):
        raise NotImplementedError("LTC is implemented in a later stage")
    else:
        raise ValueError(
            f"Unknown model name: '{name}'. Supported model encoder(s): 'gru'."
        )

    model = SessionRecommender(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        encoder=encoder,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pad_idx=pad_idx,
    )
    return model


__all__ = [
    "SequenceEncoder",
    "SessionRecommender",
    "GRUEncoder",
    "build_model",
]
