"""GRU sequence encoder implementation."""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base import SequenceEncoder


class GRUEncoder(SequenceEncoder):
    """GRU-based sequence encoder for session recommendation."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        use_time_deltas: bool = False,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_time_deltas = use_time_deltas

        input_dim = embed_dim + (1 if use_time_deltas else 0)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Initialize weights matching N(0, 0.1) standard
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        emb: torch.Tensor,
        lengths: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through GRU encoder.

        Args:
            emb: Item sequence embeddings of shape [B, T, embed_dim].
            lengths: Sequence length for each session, shape [B].
            delta_t: Optional normalized time deltas of shape [B, T] or None.

        Returns:
            Tensor of shape [B, hidden_dim] at sequence index lengths - 1.
        """
        if self.use_time_deltas:
            if delta_t is None:
                raise ValueError("delta_t is required when use_time_deltas=True")
            # Concatenate normalized time delta as an extra feature dimension
            # delta_t shape: [B, T] -> [B, T, 1]
            dt_feat = delta_t.unsqueeze(-1)
            x = torch.cat([emb, dt_feat], dim=-1)
        else:
            # Item-only mode: MUST NOT use delta_t at all (Invariant E1)
            x = emb

        # Unidirectional GRU over trimmed sequence [B, T, input_dim]
        output, _ = self.gru(x)  # output: [B, T, hidden_dim]

        # Extract output at the last valid session position (lengths - 1)
        batch_size = emb.size(0)
        batch_idx = torch.arange(batch_size, device=emb.device)
        last_idx = lengths - 1

        last_output = output[batch_idx, last_idx]  # [B, hidden_dim]
        return last_output
