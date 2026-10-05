"""Base sequence encoder interface and recommender model wrapper."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

import torch
import torch.nn as nn


class SequenceEncoder(nn.Module, ABC):
    """Abstract base class for sequence encoders in session recommendation."""

    @abstractmethod
    def forward(
        self,
        emb: torch.Tensor,
        lengths: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process sequence embeddings and extract representation at last valid position.

        Args:
            emb: Embedded item sequence of shape [B, T, embed_dim].
            lengths: Valid sequence length for each batch item, shape [B].
            delta_t: Optional normalized time deltas of shape [B, T] or None.

        Returns:
            Tensor of shape [B, hidden_dim] representing session embedding.
        """
        pass


class SessionRecommender(nn.Module):
    """Session-based recommender model with pluggable sequence encoder."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder: SequenceEncoder,
        hidden_dim: int,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx

        # Item Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # N(0, 0.1) initialization, ensuring padding index is 0
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()

        self.emb_drop = nn.Dropout(dropout)
        self.encoder = encoder
        self.enc_drop = nn.Dropout(dropout)

        # Output linear layer to all item logits
        self.head = nn.Linear(hidden_dim, vocab_size)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.head.bias)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass over a batch dict.

        Args:
            batch: Dict containing:
                - 'inputs': Tensor [B, T] (item IDs)
                - 'lengths': Tensor [B] (sequence lengths)
                - 'delta_t': Optional Tensor [B, T] or None

        Returns:
            Logits tensor of shape [B, vocab_size] with PAD logit (idx=0) masked.
            - Masked with -1e9 during training to prevent NaN/overflow issues in softmax/loss.
            - Masked with -inf during evaluation so PAD index is never selected.
        """
        inputs = batch["inputs"]
        lengths = batch["lengths"]
        delta_t = batch.get("delta_t", None)

        emb = self.emb_drop(self.embedding(inputs))
        enc_out = self.encoder(emb, lengths, delta_t)
        enc_out = self.enc_drop(enc_out)
        logits = self.head(enc_out)

        # Mask PAD logit (index 0)
        mask_val = -1e9 if self.training else float("-inf")
        logits[:, self.pad_idx] = mask_val

        return logits
