"""Focused tests for the distraction-audio GRU baseline pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from driver_state.models.unimodal.train_audio_baseline import (  # noqa: E402
    AudioGRUBaseline,
    load_feature_items,
    select_valid_rows,
)


class _IdentityGRU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x, None


def _feature_sample(root: Path, sample_id: str, shape: tuple[int, int]) -> dict[str, str]:
    path = root / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    length, dim = shape
    np.savez(
        path,
        x=np.zeros(shape, dtype=np.float32),
        time_s=np.linspace(1.0, 9.0, length, dtype=np.float64),
        valid_mask=np.ones(length, dtype=bool),
        support_s=np.array([[0.0, 10.0]] * length, dtype=np.float64),
        observed_fraction=np.ones(length, dtype=np.float32),
    )
    return {
        "sample_id": sample_id,
        "subject_id": "P01",
        "label_id": "0",
        "feature_path": path.name,
        "feature_shape": json.dumps(list(shape)),
        "feature_dtype": "float32",
    }


def test_last_pool_uses_last_valid_token_for_internal_gap() -> None:
    model = AudioGRUBaseline(input_dim=1, hidden_dim=1)
    model.proj = torch.nn.Identity()
    model.gru = _IdentityGRU()
    model.dropout = torch.nn.Identity()
    model.head = torch.nn.Identity()
    x = torch.tensor([[[1.0], [99.0], [3.0]]])
    mask = torch.tensor([[True, False, True]])

    pooled = model(x, mask)["logits"]

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0, 3.0]]))


def test_last_pool_rejects_all_invalid_tokens() -> None:
    model = AudioGRUBaseline(input_dim=1, hidden_dim=1)
    model.proj = torch.nn.Identity()
    model.gru = _IdentityGRU()
    model.dropout = torch.nn.Identity()
    model.head = torch.nn.Identity()
    x = torch.tensor([[[1.0], [2.0]]])
    mask = torch.tensor([[False, False]])

    with pytest.raises(ValueError, match="at least one valid token"):
        model(x, mask)


def test_select_valid_rows_filters_and_records_invalid_rows() -> None:
    valid = {"sample_id": "01_P01_1", "valid": "true", "feature_path": "a.npz", "error": ""}
    invalid = {"sample_id": "01_P01_2", "valid": "false", "error": "SAMPLE_RATE_DEVIATION"}

    selected, rejected = select_valid_rows([valid, invalid])

    assert selected == [valid]
    assert rejected == [{"sample_id": "01_P01_2", "error": "SAMPLE_RATE_DEVIATION"}]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"sample_id": "01_P01_1", "valid": "true", "feature_path": "a.npz",
             "error": "QC_FAILED"},
            "valid=true",
        ),
        (
            {"sample_id": "01_P01_1", "valid": "true", "feature_path": "", "error": ""},
            "no feature_path",
        ),
    ],
)
def test_select_valid_rows_rejects_contradictory_rows(
        row: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        select_valid_rows([row])


def test_load_feature_items_rejects_dimension_mismatch(tmp_path: Path) -> None:
    first = _feature_sample(tmp_path, "first", (3, 4))
    second = _feature_sample(tmp_path, "second", (3, 5))

    with pytest.raises(ValueError, match="does not match"):
        load_feature_items([first, second], tmp_path)


def test_load_feature_items_rejects_dtype_mismatch(tmp_path: Path) -> None:
    sample = _feature_sample(tmp_path, "bad_dtype", (3, 4))
    path = tmp_path / sample["feature_path"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["x"] = arrays["x"].astype(np.float64)
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="dtype"):
        load_feature_items([sample], tmp_path)
