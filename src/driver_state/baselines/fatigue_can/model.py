"""Mask-safe single-layer GRU baseline for 30-second CAN windows."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class CanGruBaseline(nn.Module):
    """Project valid CAN tokens, encode them with one GRU, and classify.

    Internal invalid positions are removed before the recurrent layer rather
    than being treated as a simple sequence length. This makes predictions
    independent of the numeric values stored at masked positions.
    """

    def __init__(
        self,
        *,
        input_dim: int = 9,
        projection_dim: int = 64,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if min(input_dim, projection_dim, hidden_dim, num_classes) <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.input_dim = int(input_dim)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.projection_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=self.projection_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        if "can" not in inputs:
            raise KeyError("CanGruBaseline requires inputs['can']")
        can = inputs["can"]
        x = can["x"]
        valid_mask = can["valid_mask"]
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"CAN x must have shape [B,T,{self.input_dim}]")
        if valid_mask.dtype != torch.bool or valid_mask.shape != x.shape[:2]:
            raise ValueError("valid_mask must be bool with shape [B,T]")
        lengths = valid_mask.sum(dim=1)
        if (lengths == 0).any():
            raise ValueError("all-invalid CAN samples are forbidden")

        projected = self.projection(x)
        valid_sequences = [
            projected[index, valid_mask[index]]
            for index in range(projected.shape[0])
        ]
        padded = pad_sequence(valid_sequences, batch_first=True)
        packed = pack_padded_sequence(
            padded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        embedding = hidden[-1]
        logits = self.classifier(embedding)
        return {"logits": logits, "embedding": embedding}
