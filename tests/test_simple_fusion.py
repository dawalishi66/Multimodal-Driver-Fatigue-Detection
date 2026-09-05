from __future__ import annotations

import pytest
import torch

from driver_state.models.simple_fusion import SimpleFusion


def _make_stream(
    batch_size: int,
    steps: int,
    feature_dim: int,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    x = torch.arange(
        batch_size * steps * feature_dim, dtype=torch.float32
    ).reshape(batch_size, steps, feature_dim) / 100.0
    time_s = torch.arange(steps, dtype=torch.float64).repeat(batch_size, 1)
    return {
        "x": x,
        "valid_mask": valid_mask.clone(),
        "time_s": time_s,
    }


def _make_inputs(
    modalities: tuple[str, str] = ("video", "can"),
    feature_dims: tuple[int, int] = (7, 11),
    masks: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    if masks is None:
        masks = (
            torch.tensor(
                [
                    [True, False, True, False, True],
                    [True, True, False, False, False],
                    [False, True, False, True, False],
                ],
                dtype=torch.bool,
            ),
            torch.tensor(
                [
                    [True, False, True],
                    [True, True, True],
                    [False, True, False],
                ],
                dtype=torch.bool,
            ),
        )

    batch_size = masks[0].shape[0]
    return {
        modality: _make_stream(batch_size, mask.shape[1], feature_dim, mask)
        for modality, feature_dim, mask in zip(modalities, feature_dims, masks)
    }


def _make_model(
    modalities: tuple[str, str] = ("video", "can"),
    feature_dims: tuple[int, int] = (7, 11),
    num_classes: int = 3,
) -> SimpleFusion:
    return SimpleFusion(
        modalities=modalities,
        input_dims=dict(zip(modalities, feature_dims)),
        num_classes=num_classes,
        projection_dim=5,
        classifier_hidden_dim=4,
        dropout=0.2,
    )


def _clone_inputs(
    inputs: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        modality: {name: value.clone() for name, value in stream.items()}
        for modality, stream in inputs.items()
    }


def _append_invalid_padding(
    inputs: dict[str, dict[str, torch.Tensor]],
    padding_steps: int,
) -> dict[str, dict[str, torch.Tensor]]:
    padded = _clone_inputs(inputs)
    for stream in padded.values():
        x = stream["x"]
        batch_size, _, feature_dim = x.shape
        stream["x"] = torch.cat(
            [
                x,
                torch.full(
                    (batch_size, padding_steps, feature_dim),
                    torch.finfo(x.dtype).max,
                    dtype=x.dtype,
                ),
            ],
            dim=1,
        )
        stream["valid_mask"] = torch.cat(
            [
                stream["valid_mask"],
                torch.zeros(
                    batch_size, padding_steps, dtype=torch.bool
                ),
            ],
            dim=1,
        )
        time_s = stream["time_s"]
        stream["time_s"] = torch.cat(
            [
                time_s,
                torch.full(
                    (batch_size, padding_steps),
                    -999.0,
                    dtype=time_s.dtype,
                ),
            ],
            dim=1,
        )
    return padded


def test_fatigue_returns_three_class_logits_for_two_modalities():
    model = _make_model()

    output = model(_make_inputs())

    assert set(output) == {"logits"}
    assert output["logits"].shape == (3, 3)
    assert torch.isfinite(output["logits"]).all()


def test_distraction_returns_nine_class_logits():
    model = _make_model(
        modalities=("video", "audio"),
        feature_dims=(7, 13),
        num_classes=9,
    )
    masks = (
        torch.tensor([[True, False, True], [True, True, False]], dtype=torch.bool),
        torch.tensor(
            [[True, False, True, False], [True, True, True, False]],
            dtype=torch.bool,
        ),
    )

    output = model(_make_inputs(("video", "audio"), (7, 13), masks))

    assert output["logits"].shape == (2, 9)
    assert torch.isfinite(output["logits"]).all()


def test_supports_different_sequence_lengths_and_per_sample_valid_lengths():
    model = _make_model()
    inputs = _make_inputs()

    assert inputs["video"]["x"].shape[1] != inputs["can"]["x"].shape[1]
    assert inputs["video"]["valid_mask"].sum(dim=1).tolist() == [3, 2, 2]
    assert inputs["can"]["valid_mask"].sum(dim=1).tolist() == [2, 3, 1]
    output = model(inputs)

    assert output["logits"].shape == (3, 3)


def test_supports_single_valid_token_per_modality():
    model = _make_model()
    masks = (
        torch.ones(2, 1, dtype=torch.bool),
        torch.ones(2, 1, dtype=torch.bool),
    )

    output = model(_make_inputs(feature_dims=(7, 11), masks=masks))

    assert output["logits"].shape == (2, 3)
    assert torch.isfinite(output["logits"]).all()


def test_internal_invalid_tokens_are_supported():
    model = _make_model().eval()
    inputs = _make_inputs()

    output = model(inputs)["logits"]

    assert output.shape == (3, 3)
    assert torch.isfinite(output).all()


def test_invalid_token_values_do_not_change_eval_logits():
    model = _make_model().eval()
    inputs = _make_inputs()
    mutated = _clone_inputs(inputs)

    for stream in mutated.values():
        invalid = ~stream["valid_mask"]
        replacement = torch.finfo(stream["x"].dtype).max
        stream["x"][invalid] = replacement

    baseline_logits = model(inputs)["logits"]
    mutated_logits = model(mutated)["logits"]

    torch.testing.assert_close(baseline_logits, mutated_logits)


def test_extra_tail_invalid_padding_does_not_change_eval_logits():
    model = _make_model().eval()
    inputs = _make_inputs()
    padded = _append_invalid_padding(inputs, padding_steps=3)

    baseline_logits = model(inputs)["logits"]
    padded_logits = model(padded)["logits"]

    torch.testing.assert_close(baseline_logits, padded_logits)


@pytest.mark.parametrize("modality, row", [("video", 0), ("can", 1)])
def test_rejects_a_batch_sample_with_no_valid_tokens(modality: str, row: int):
    model = _make_model()
    inputs = _make_inputs()
    inputs[modality]["valid_mask"][row] = False

    with pytest.raises(ValueError):
        model(inputs)


def test_rejects_non_mapping_inputs():
    model = _make_model()

    with pytest.raises((TypeError, ValueError)):
        model([])


def test_rejects_missing_modality():
    model = _make_model()
    inputs = _make_inputs()
    del inputs["can"]

    with pytest.raises(ValueError):
        model(inputs)


def test_rejects_extra_modality():
    model = _make_model()
    inputs = _make_inputs()
    inputs["audio"] = _make_stream(
        3,
        2,
        13,
        torch.ones(3, 2, dtype=torch.bool),
    )

    with pytest.raises(ValueError):
        model(inputs)


def test_rejects_missing_stream_field():
    model = _make_model()
    inputs = _make_inputs()
    del inputs["video"]["time_s"]

    with pytest.raises(ValueError):
        model(inputs)


def test_rejects_wrong_feature_dimension():
    model = _make_model()
    inputs = _make_inputs()
    inputs["video"]["x"] = torch.zeros(3, 5, 6, dtype=torch.float32)

    with pytest.raises(ValueError):
        model(inputs)


def test_rejects_mismatched_batch_sizes_between_modalities():
    model = _make_model()
    inputs = _make_inputs()
    inputs["can"] = _make_stream(
        2,
        3,
        11,
        torch.ones(2, 3, dtype=torch.bool),
    )

    with pytest.raises(ValueError):
        model(inputs)


def test_rejects_non_bool_valid_mask():
    model = _make_model()
    inputs = _make_inputs()
    inputs["video"]["valid_mask"] = inputs["video"]["valid_mask"].to(
        torch.int64
    )

    with pytest.raises(ValueError):
        model(inputs)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_x(bad_value: float):
    model = _make_model()
    inputs = _make_inputs()
    inputs["video"]["x"][0, 0, 0] = bad_value

    with pytest.raises(ValueError):
        model(inputs)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_time_s(bad_value: float):
    model = _make_model()
    inputs = _make_inputs()
    inputs["can"]["time_s"][0, 0] = bad_value

    with pytest.raises(ValueError):
        model(inputs)


@pytest.mark.parametrize(
    "updates",
    [
        {"modalities": ("video",)},
        {"modalities": ("video", "video")},
        {"input_dims": {"video": 7}},
        {"input_dims": {"video": 7, "can": 0}},
        {"num_classes": 1},
        {"projection_dim": 0},
        {"classifier_hidden_dim": 0},
        {"dropout": -0.1},
        {"dropout": 1.1},
        {"dropout": float("nan")},
        {"dropout": float("inf")},
    ],
)
def test_constructor_rejects_invalid_configuration(updates):
    kwargs = {
        "modalities": ("video", "can"),
        "input_dims": {"video": 7, "can": 11},
        "num_classes": 3,
        "projection_dim": 5,
        "classifier_hidden_dim": 4,
        "dropout": 0.2,
    }
    kwargs.update(updates)

    with pytest.raises((TypeError, ValueError)):
        SimpleFusion(**kwargs)


def test_forward_logits_and_gradients_are_finite():
    model = _make_model().train()
    inputs = _make_inputs()
    labels = torch.tensor([0, 2, 1], dtype=torch.long)

    logits = model(inputs)["logits"]
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
