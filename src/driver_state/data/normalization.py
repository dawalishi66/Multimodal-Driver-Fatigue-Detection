"""Train-only, mask-aware feature normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class MaskedStandardizer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    valid_token_count: int
    zero_variance_indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    fitted_split: str
    source_manifest_sha256: str
    version: str = "masked_zscore_v1"

    @property
    def feature_dim(self) -> int:
        return len(self.mean)

    @classmethod
    def fit_can_dataset(
        cls,
        dataset: Any,
        *,
        feature_names: Sequence[str],
        variance_epsilon: float = 1e-12,
    ) -> "MaskedStandardizer":
        if getattr(dataset, "split", None) != "train":
            raise ValueError("normalization may only be fitted on the train split")
        names = tuple(feature_names)
        if not names:
            raise ValueError("feature_names must not be empty")
        sums = np.zeros(len(names), dtype=np.float64)
        squared_sums = np.zeros(len(names), dtype=np.float64)
        valid_token_count = 0
        for index in range(len(dataset)):
            can = dataset[index]["inputs"]["can"]
            x = can["x"].numpy()
            mask = can["valid_mask"].numpy()
            valid = x[mask].astype(np.float64, copy=False)
            if valid.shape[1] != len(names):
                raise ValueError("feature_names do not match the CAN feature dimension")
            sums += valid.sum(axis=0, dtype=np.float64)
            squared_sums += np.square(valid, dtype=np.float64).sum(
                axis=0, dtype=np.float64
            )
            valid_token_count += int(valid.shape[0])
        if valid_token_count == 0:
            raise ValueError("cannot fit normalization without valid training tokens")
        mean = sums / valid_token_count
        variance = np.maximum(
            squared_sums / valid_token_count - np.square(mean), 0.0
        )
        zero_variance = np.flatnonzero(variance <= variance_epsilon)
        scale = np.sqrt(variance)
        scale[zero_variance] = 1.0
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("normalization statistics contain NaN or Inf")
        return cls(
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
            valid_token_count=valid_token_count,
            zero_variance_indices=tuple(int(value) for value in zero_variance),
            feature_names=names,
            fitted_split="train",
            source_manifest_sha256=str(dataset.manifest_sha256),
        )

    def transform_tensor(
        self, x: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim not in (2, 3):
            raise ValueError("x must have shape [T,D] or [B,T,D]")
        if valid_mask.shape != x.shape[:-1] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool and match x without its last axis")
        if x.shape[-1] != self.feature_dim:
            raise ValueError("x feature dimension does not match the standardizer")
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        scale = torch.as_tensor(self.scale, dtype=x.dtype, device=x.device)
        transformed = (x - mean) / scale
        transformed = transformed.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(transformed).all():
            raise ValueError("standardized features contain NaN or Inf")
        return transformed

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "method": "zscore",
            "fitted_split": self.fitted_split,
            "mask_policy": "fit_valid_tokens_only_and_zero_invalid_after_transform",
            "source_manifest_sha256": self.source_manifest_sha256,
            "valid_token_count": self.valid_token_count,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "zero_variance_indices": list(self.zero_variance_indices),
        }

    def save_json(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "MaskedStandardizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != "masked_zscore_v1":
            raise ValueError("unsupported normalization version")
        if data.get("fitted_split") != "train":
            raise ValueError("normalization state was not fitted on train")
        return cls(
            mean=tuple(float(value) for value in data["mean"]),
            scale=tuple(float(value) for value in data["scale"]),
            valid_token_count=int(data["valid_token_count"]),
            zero_variance_indices=tuple(
                int(value) for value in data["zero_variance_indices"]
            ),
            feature_names=tuple(data["feature_names"]),
            fitted_split=data["fitted_split"],
            source_manifest_sha256=data["source_manifest_sha256"],
            version=data["version"],
        )
