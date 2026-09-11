"""Small, deterministic training primitives shared by classification baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EarlyStoppingTracker:
    """Track a maximized validation metric while keeping the earlier tie."""

    patience: int
    best_metric: float = -math.inf
    best_epoch: int | None = None
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")

    def update(self, *, epoch: int, metric: float) -> bool:
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        if not math.isfinite(metric):
            raise ValueError("selection metric must be finite")
        if metric > self.best_metric:
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience

