from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import torch
from torch import Tensor, nn


class SimpleFusion(nn.Module):
    """A two-stream token-wise projection and mean-pooling classifier."""

    def __init__(
        self,
        modalities: tuple[str, str],
        input_dims: dict[str, int],
        num_classes: int,
        projection_dim: int,
        classifier_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self._validate_configuration(
            modalities,
            input_dims,
            num_classes,
            projection_dim,
            classifier_hidden_dim,
            dropout,
        )

        self.modalities = modalities
        self.input_dims = dict(input_dims)

        self.projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(self.input_dims[modality], projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for modality in modalities
        })
        self.classifier = nn.Sequential(
            nn.Linear(2 * projection_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    @staticmethod
    def _validate_configuration(
        modalities: tuple[str, str],
        input_dims: dict[str, int],
        num_classes: int,
        projection_dim: int,
        classifier_hidden_dim: int,
        dropout: float,
    ) -> None:
        if not isinstance(modalities, tuple) or len(modalities) != 2:
            raise ValueError("modalities must be a tuple containing exactly two names")
        if any(not isinstance(modality, str) or not modality for modality in modalities):
            raise ValueError("each modality must be a non-empty string")
        if modalities[0] == modalities[1]:
            raise ValueError("modalities must be different")
        if any("." in modality for modality in modalities):
            raise ValueError("modality names must not contain '.'")

        if not isinstance(input_dims, Mapping):
            raise ValueError("input_dims must be a mapping")
        if set(input_dims) != set(modalities):
            raise ValueError("input_dims must exactly cover modalities")
        for modality in modalities:
            dimension = input_dims[modality]
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise ValueError(f"input dimension for {modality!r} must be positive")

        for name, value in (
            ("num_classes", num_classes),
            ("projection_dim", projection_dim),
            ("classifier_hidden_dim", classifier_hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")

        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(float(dropout))
            or not 0.0 <= float(dropout) <= 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1]")

    def forward(self, inputs: Mapping[str, Mapping[str, Tensor]]) -> dict[str, Tensor]:
        validated_inputs = self._validate_inputs(inputs)
        pooled = [
            self._pool_tokens(x, valid_mask, self.projections[modality])
            for modality, (x, valid_mask) in zip(self.modalities, validated_inputs)
        ]
        logits = self.classifier(torch.cat(pooled, dim=-1))
        return {"logits": logits}

    def _validate_inputs(
        self,
        inputs: Mapping[str, Mapping[str, Tensor]],
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if not isinstance(inputs, Mapping):
            raise ValueError("inputs must be a mapping")
        if set(inputs) != set(self.modalities):
            raise ValueError("inputs must contain exactly the declared modalities")

        validated: list[tuple[Tensor, Tensor]] = []
        batch_size: int | None = None
        for modality in self.modalities:
            stream = inputs[modality]
            if not isinstance(stream, Mapping):
                raise ValueError(f"input for modality {modality!r} must be a mapping")
            missing = {"x", "valid_mask", "time_s"} - set(stream)
            if missing:
                raise ValueError(
                    f"input for modality {modality!r} is missing {sorted(missing)}"
                )

            x = stream["x"]
            valid_mask = stream["valid_mask"]
            time_s = stream["time_s"]
            if not isinstance(x, Tensor):
                raise ValueError(f"{modality}.x must be a Tensor")
            if not isinstance(valid_mask, Tensor):
                raise ValueError(f"{modality}.valid_mask must be a Tensor")
            if not isinstance(time_s, Tensor):
                raise ValueError(f"{modality}.time_s must be a Tensor")
            if x.ndim != 3:
                raise ValueError(f"{modality}.x must have shape [B, T, D]")
            if valid_mask.ndim != 2:
                raise ValueError(f"{modality}.valid_mask must have shape [B, T]")
            if time_s.ndim != 2:
                raise ValueError(f"{modality}.time_s must have shape [B, T]")
            if not torch.is_floating_point(x):
                raise ValueError(f"{modality}.x must be a floating Tensor")
            if valid_mask.dtype != torch.bool:
                raise ValueError(f"{modality}.valid_mask must have dtype torch.bool")
            if not torch.is_floating_point(time_s):
                raise ValueError(f"{modality}.time_s must be a floating Tensor")

            current_batch, current_steps, current_dim = x.shape
            if valid_mask.shape != (current_batch, current_steps):
                raise ValueError(f"{modality}.valid_mask must match x B and T")
            if time_s.shape != (current_batch, current_steps):
                raise ValueError(f"{modality}.time_s must match x B and T")
            if current_dim != self.input_dims[modality]:
                raise ValueError(
                    f"{modality}.x feature dimension must be "
                    f"{self.input_dims[modality]}"
                )
            if x.device != valid_mask.device or x.device != time_s.device:
                raise ValueError(f"{modality} tensors must be on the same device")
            if not torch.isfinite(x).all().item():
                raise ValueError(f"{modality}.x must contain only finite values")
            if not torch.isfinite(time_s).all().item():
                raise ValueError(f"{modality}.time_s must contain only finite values")
            if not valid_mask.any(dim=1).all().item():
                raise ValueError(
                    f"every {modality} batch sample must have a valid token"
                )
            if batch_size is None:
                batch_size = current_batch
            elif current_batch != batch_size:
                raise ValueError("modalities must have the same batch size")

            validated.append((x, valid_mask))

        return validated[0], validated[1]

    @staticmethod
    def _pool_tokens(
        x: Tensor,
        valid_mask: Tensor,
        projection: nn.Module,
    ) -> Tensor:
        valid = valid_mask.unsqueeze(-1)
        safe_x = torch.where(valid, x, torch.zeros_like(x))
        projection_dtype = next(projection.parameters()).dtype
        if safe_x.dtype != projection_dtype:
            safe_x = safe_x.to(dtype=projection_dtype)
        projected = projection(safe_x)
        projected = projected.masked_fill(~valid, 0.0)
        valid_count = valid_mask.sum(dim=1, keepdim=True).to(dtype=projected.dtype)
        return projected.sum(dim=1) / valid_count
