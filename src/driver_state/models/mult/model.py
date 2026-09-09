"""A true two-modal MulT adapter for the v0.2 driver-state interface.

The architecture follows Tsai et al., *Multimodal Transformer for Unaligned
Multimodal Language Sequences* (ACL 2019), with reference code at
https://github.com/yaohungt/Multimodal-Transformer.

This implementation is a two-modal PyTorch 2.14 adaptation.  It does not
copy the complete upstream repository or its legacy attention implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import torch
from torch import Tensor, nn


def _masked_positions(states: Tensor, valid_mask: Tensor) -> Tensor:
    """Keep invalid token states exactly zero without changing valid states."""

    return states.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


def _sinusoidal_positions(
    length: int,
    embedding_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create dynamic 1-based sinusoidal sequence positions.

    The computation is performed in float32 and converted to the model
    dtype at the boundary.  Positions depend only on sequence slots, never
    on feature values, valid masks, or real timestamps.
    """

    position = torch.arange(
        1,
        length + 1,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1)
    even_indices = torch.arange(
        0,
        embedding_dim,
        2,
        device=device,
        dtype=torch.float32,
    )
    scale = -math.log(10000.0) / embedding_dim
    angles = position * torch.exp(even_indices * scale)

    encoding = torch.zeros(
        length,
        embedding_dim,
        device=device,
        dtype=torch.float32,
    )
    encoding[:, 0::2] = torch.sin(angles)
    odd_count = encoding[:, 1::2].shape[1]
    if odd_count:
        encoding[:, 1::2] = torch.cos(angles[:, :odd_count])
    return encoding.to(dtype=dtype)


class _MaskedMulTBlock(nn.Module):
    """Pre-norm attention/FFN block with explicit query-state masking."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: Tensor,
        source: Tensor,
        target_valid_mask: Tensor,
        source_valid_mask: Tensor,
    ) -> Tensor:
        target = _masked_positions(target, target_valid_mask)
        source = _masked_positions(source, source_valid_mask)

        normalized_target = self.attention_norm(target)
        normalized_source = self.attention_norm(source)
        attended, _ = self.attention(
            normalized_target,
            normalized_source,
            normalized_source,
            key_padding_mask=~source_valid_mask,
            need_weights=False,
            is_causal=False,
        )
        target = target + self.attention_dropout(attended)
        target = _masked_positions(target, target_valid_mask)

        feed_forward = self.ffn(self.ffn_norm(target))
        target = target + self.ffn_dropout(feed_forward)
        return _masked_positions(target, target_valid_mask)


class DualModalMulT(nn.Module):
    """A configurable two-modal MulT classifier.

    ``modalities`` gives the ordered target streams A and B.  Each stream is
    a mapping containing ``x``, ``valid_mask`` and ``time_s``.  The model
    returns raw classifier logits only.
    """

    def __init__(
        self,
        modalities: tuple[str, str],
        input_dims: Mapping[str, int],
        num_classes: int,
        d_model: int = 30,
        num_heads: int = 5,
        cross_layers: int = 5,
        memory_layers: int = 5,
        dropout: float = 0.0,
        enable_a_from_b: bool = True,
        enable_b_from_a: bool = True,
        causal_attention: bool = False,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            modalities=modalities,
            input_dims=input_dims,
            num_classes=num_classes,
            d_model=d_model,
            num_heads=num_heads,
            cross_layers=cross_layers,
            memory_layers=memory_layers,
            dropout=dropout,
            enable_a_from_b=enable_a_from_b,
            enable_b_from_a=enable_b_from_a,
            causal_attention=causal_attention,
        )

        self.modalities = modalities
        self.input_dims = dict(input_dims)
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.cross_layers = cross_layers
        self.memory_layers = memory_layers
        self.dropout = float(dropout)
        self.enable_a_from_b = enable_a_from_b
        self.enable_b_from_a = enable_b_from_a
        self.causal_attention = False

        self.projections = nn.ModuleDict(
            {
                modality: nn.Linear(self.input_dims[modality], d_model, bias=False)
                for modality in modalities
            }
        )

        self.cross_a_from_b = nn.ModuleList(
            [
                _MaskedMulTBlock(d_model, num_heads, self.dropout)
                for _ in range(cross_layers)
            ]
            if enable_a_from_b
            else []
        )
        self.cross_a_final_norm: nn.Module = (
            nn.LayerNorm(d_model) if enable_a_from_b else nn.Identity()
        )
        self.cross_b_from_a = nn.ModuleList(
            [
                _MaskedMulTBlock(d_model, num_heads, self.dropout)
                for _ in range(cross_layers)
            ]
            if enable_b_from_a
            else []
        )
        self.cross_b_final_norm: nn.Module = (
            nn.LayerNorm(d_model) if enable_b_from_a else nn.Identity()
        )
        self.memory_a = nn.ModuleList(
            [
                _MaskedMulTBlock(d_model, num_heads, self.dropout)
                for _ in range(memory_layers)
            ]
        )
        self.memory_a_final_norm = nn.LayerNorm(d_model)
        self.memory_b = nn.ModuleList(
            [
                _MaskedMulTBlock(d_model, num_heads, self.dropout)
                for _ in range(memory_layers)
            ]
        )
        self.memory_b_final_norm = nn.LayerNorm(d_model)

        combined_dim = 2 * d_model
        self.residual_mlp = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(combined_dim, combined_dim),
        )
        self.classifier = nn.Linear(combined_dim, num_classes)

    @staticmethod
    def _validate_configuration(
        *,
        modalities: tuple[str, str],
        input_dims: Mapping[str, int],
        num_classes: int,
        d_model: int,
        num_heads: int,
        cross_layers: int,
        memory_layers: int,
        dropout: float,
        enable_a_from_b: bool,
        enable_b_from_a: bool,
        causal_attention: bool,
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
            ("d_model", d_model),
            ("num_heads", num_heads),
            ("cross_layers", cross_layers),
            ("memory_layers", memory_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(float(dropout))
            or not 0.0 <= float(dropout) <= 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1]")
        if not isinstance(enable_a_from_b, bool):
            raise ValueError("enable_a_from_b must be bool")
        if not isinstance(enable_b_from_a, bool):
            raise ValueError("enable_b_from_a must be bool")
        if not isinstance(causal_attention, bool):
            raise ValueError("causal_attention must be bool")
        if causal_attention:
            raise ValueError("causal attention is disabled in the first version")

    def forward(self, inputs: Mapping[str, Mapping[str, Tensor]]) -> dict[str, Tensor]:
        validated = self._validate_inputs(inputs)
        (x_a, mask_a), (x_b, mask_b) = validated

        base_a = self._encode_stream(x_a, mask_a, self.projections[self.modalities[0]])
        base_b = self._encode_stream(x_b, mask_b, self.projections[self.modalities[1]])

        target_a = base_a
        if self.enable_a_from_b:
            for block in self.cross_a_from_b:
                target_a = block(target_a, base_b, mask_a, mask_b)
        target_a = _masked_positions(self.cross_a_final_norm(target_a), mask_a)

        target_b = base_b
        if self.enable_b_from_a:
            for block in self.cross_b_from_a:
                target_b = block(target_b, base_a, mask_b, mask_a)
        target_b = _masked_positions(self.cross_b_final_norm(target_b), mask_b)

        for block in self.memory_a:
            target_a = block(target_a, target_a, mask_a, mask_a)
        target_a = _masked_positions(self.memory_a_final_norm(target_a), mask_a)
        for block in self.memory_b:
            target_b = block(target_b, target_b, mask_b, mask_b)
        target_b = _masked_positions(self.memory_b_final_norm(target_b), mask_b)

        last_a = self._last_valid(target_a, mask_a)
        last_b = self._last_valid(target_b, mask_b)
        fused = torch.cat((last_a, last_b), dim=-1)
        fused = fused + self.residual_mlp(fused)
        logits = self.classifier(fused)
        return {"logits": logits}

    def _validate_inputs(
        self,
        inputs: Mapping[str, Mapping[str, Tensor]],
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if not isinstance(inputs, Mapping):
            raise ValueError("inputs must be a mapping")
        if set(inputs) != set(self.modalities):
            raise ValueError("inputs must contain exactly the declared modalities")

        model_device = next(self.parameters()).device
        validated: list[tuple[Tensor, Tensor]] = []
        batch_size: int | None = None
        for modality in self.modalities:
            stream = inputs[modality]
            if not isinstance(stream, Mapping):
                raise ValueError(f"input for modality {modality!r} must be a mapping")
            if set(stream) != {"x", "valid_mask", "time_s"}:
                raise ValueError(
                    f"input for modality {modality!r} must contain exactly "
                    "x, valid_mask, and time_s"
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
            if current_batch == 0:
                raise ValueError("batch size must be greater than zero")
            if current_steps == 0:
                raise ValueError(f"{modality} sequence length must be greater than zero")
            if valid_mask.shape != (current_batch, current_steps):
                raise ValueError(f"{modality}.valid_mask must match x B and T")
            if time_s.shape != (current_batch, current_steps):
                raise ValueError(f"{modality}.time_s must match x B and T")
            if current_dim != self.input_dims[modality]:
                raise ValueError(
                    f"{modality}.x feature dimension must be {self.input_dims[modality]}"
                )
            if x.device != valid_mask.device or x.device != time_s.device:
                raise ValueError(f"{modality} tensors must be on the same device")
            if x.device != model_device:
                raise ValueError("inputs must be on the same device as the model")
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
    def _encode_stream(
        x: Tensor,
        valid_mask: Tensor,
        projection: nn.Linear,
    ) -> Tensor:
        valid = valid_mask.unsqueeze(-1)
        safe_x = x.masked_fill(~valid, 0.0)
        projection_dtype = projection.weight.dtype
        if safe_x.dtype != projection_dtype:
            safe_x = safe_x.to(dtype=projection_dtype)
            if not torch.isfinite(safe_x).all().item():
                raise ValueError("valid input values are not representable in model dtype")
        projected = projection(safe_x)
        projected = projected * math.sqrt(projection.out_features)
        position = _sinusoidal_positions(
            x.shape[1],
            projection.out_features,
            device=x.device,
            dtype=projected.dtype,
        ).unsqueeze(0)
        encoded = projected + position
        return _masked_positions(encoded, valid_mask)

    @staticmethod
    def _last_valid(states: Tensor, valid_mask: Tensor) -> Tensor:
        positions = torch.arange(states.shape[1], device=states.device).expand(
            states.shape[0], -1
        )
        last_indices = positions.masked_fill(~valid_mask, -1).amax(dim=1)
        batch_indices = torch.arange(states.shape[0], device=states.device)
        return states[batch_indices, last_indices]
