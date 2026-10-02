from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from driver_state.models.mult import DualModalMulT


def _make_stream(
    batch_size: int,
    steps: int,
    feature_dim: int,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if valid_mask is None:
        valid_mask = torch.ones(batch_size, steps, dtype=torch.bool)
    x = torch.arange(
        batch_size * steps * feature_dim,
        dtype=torch.float32,
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
                    [True, False, True, False],
                    [True, True, False, False],
                    [False, True, False, True],
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
    return {
        modality: _make_stream(mask.shape[0], mask.shape[1], feature_dim, mask)
        for modality, feature_dim, mask in zip(modalities, feature_dims, masks)
    }


def _make_model(
    modalities: tuple[str, str] = ("video", "can"),
    feature_dims: tuple[int, int] = (7, 11),
    num_classes: int = 3,
    *,
    enable_a_from_b: bool = True,
    enable_b_from_a: bool = True,
) -> DualModalMulT:
    return DualModalMulT(
        modalities=modalities,
        input_dims=dict(zip(modalities, feature_dims)),
        num_classes=num_classes,
        d_model=10,
        num_heads=5,
        cross_layers=1,
        memory_layers=1,
        dropout=0.0,
        enable_a_from_b=enable_a_from_b,
        enable_b_from_a=enable_b_from_a,
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
    padding_steps: tuple[int, int] = (3, 2),
) -> dict[str, dict[str, torch.Tensor]]:
    padded = _clone_inputs(inputs)
    for (stream, extra_steps) in zip(padded.values(), padding_steps):
        x = stream["x"]
        batch_size, _, feature_dim = x.shape
        stream["x"] = torch.cat(
            [
                x,
                torch.full(
                    (batch_size, extra_steps, feature_dim),
                    torch.finfo(x.dtype).max,
                    dtype=x.dtype,
                ),
            ],
            dim=1,
        )
        stream["valid_mask"] = torch.cat(
            [
                stream["valid_mask"],
                torch.zeros(batch_size, extra_steps, dtype=torch.bool),
            ],
            dim=1,
        )
        time_s = stream["time_s"]
        stream["time_s"] = torch.cat(
            [
                time_s,
                torch.full(
                    (batch_size, extra_steps),
                    -999.0,
                    dtype=time_s.dtype,
                ),
            ],
            dim=1,
        )
    return padded


def _assert_finite_participating_gradients(model: DualModalMulT) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)


def test_fatigue_returns_only_three_class_raw_logits():
    model = _make_model()

    output = model(_make_inputs())

    assert set(output) == {"logits"}
    assert output["logits"].shape == (3, 3)
    assert torch.isfinite(output["logits"]).all()


def test_distraction_returns_nine_class_raw_logits():
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

    assert set(output) == {"logits"}
    assert output["logits"].shape == (2, 9)
    assert torch.isfinite(output["logits"]).all()


def test_supports_different_lengths_and_batch_specific_valid_lengths():
    model = _make_model()
    inputs = _make_inputs()

    assert inputs["video"]["x"].shape[1] != inputs["can"]["x"].shape[1]
    assert inputs["video"]["valid_mask"].sum(dim=1).tolist() == [2, 2, 2]
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


def test_supports_odd_d_model_when_divisible_by_num_heads():
    model = DualModalMulT(
        modalities=("video", "can"),
        input_dims={"video": 7, "can": 11},
        num_classes=3,
        d_model=15,
        num_heads=3,
        cross_layers=1,
        memory_layers=1,
    )

    output = model(_make_inputs())

    assert output["logits"].shape == (3, 3)
    assert torch.isfinite(output["logits"]).all()


def test_float64_features_are_converted_to_model_dtype():
    model = _make_model().eval()
    inputs = _make_inputs()
    for stream in inputs.values():
        stream["x"] = stream["x"].to(torch.float64)

    output = model(inputs)

    assert output["logits"].dtype == next(model.parameters()).dtype
    assert torch.isfinite(output["logits"]).all()


def test_internal_invalid_tokens_are_supported_without_compacting_positions():
    model = _make_model().eval()
    masks = (
        torch.tensor([[True, False, True, False]], dtype=torch.bool),
        torch.tensor([[True, True, False]], dtype=torch.bool),
    )

    output = model(_make_inputs(masks=masks))

    assert output["logits"].shape == (1, 3)
    assert torch.isfinite(output["logits"]).all()


def test_invalid_token_values_do_not_change_eval_logits():
    model = _make_model().eval()
    inputs = _make_inputs()
    mutated = _clone_inputs(inputs)

    for stream in mutated.values():
        invalid = ~stream["valid_mask"]
        stream["x"][invalid] = torch.finfo(stream["x"].dtype).max

    baseline_logits = model(inputs)["logits"]
    mutated_logits = model(mutated)["logits"]

    torch.testing.assert_close(baseline_logits, mutated_logits)


def test_extra_tail_invalid_padding_does_not_change_eval_logits():
    model = _make_model().eval()
    inputs = _make_inputs()
    padded = _append_invalid_padding(inputs)

    baseline_logits = model(inputs)["logits"]
    padded_logits = model(padded)["logits"]

    torch.testing.assert_close(baseline_logits, padded_logits)


def test_time_s_values_do_not_change_eval_logits():
    model = _make_model().eval()
    inputs = _make_inputs()
    changed_time = _clone_inputs(inputs)
    for stream in changed_time.values():
        stream["time_s"] = stream["time_s"] * -7.0 + 123.0

    baseline_logits = model(inputs)["logits"]
    changed_logits = model(changed_time)["logits"]

    torch.testing.assert_close(baseline_logits, changed_logits)


def test_last_valid_selects_slot_two_for_internal_gap():
    states = torch.tensor(
        [[[
            10.0,
        ], [
            20.0,
        ], [
            30.0,
        ], [
            40.0,
        ]]],
    )
    valid_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)

    selected = DualModalMulT._last_valid(states, valid_mask)

    torch.testing.assert_close(selected, torch.tensor([[30.0]]))


def test_all_invalid_batch_row_is_rejected_before_attention():
    model = _make_model()
    inputs = _make_inputs()
    inputs["can"]["valid_mask"][1] = False
    attention_modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.MultiheadAttention)
    ]
    assert attention_modules
    calls = 0

    def count_attention_calls(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    hooks = [module.register_forward_pre_hook(count_attention_calls) for module in attention_modules]
    try:
        with pytest.raises(ValueError):
            model(inputs)
    finally:
        for hook in hooks:
            hook.remove()

    assert calls == 0


@pytest.mark.parametrize(
    ("enable_a_from_b", "enable_b_from_a"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_all_direction_combinations_return_finite_fixed_width_logits(
    enable_a_from_b: bool,
    enable_b_from_a: bool,
):
    model = _make_model(
        enable_a_from_b=enable_a_from_b,
        enable_b_from_a=enable_b_from_a,
    )

    output = model(_make_inputs())

    assert output["logits"].shape == (3, 3)
    assert torch.isfinite(output["logits"]).all()


@pytest.mark.parametrize(
    ("enable_a_from_b", "enable_b_from_a"),
    [(True, False), (False, True), (False, False)],
)
def test_disabled_cross_directions_have_no_trainable_parameters(
    enable_a_from_b: bool,
    enable_b_from_a: bool,
):
    model = _make_model(
        enable_a_from_b=enable_a_from_b,
        enable_b_from_a=enable_b_from_a,
    )

    if enable_a_from_b:
        assert list(model.cross_a_from_b.parameters())
    else:
        assert not list(model.cross_a_from_b.parameters())
    if enable_b_from_a:
        assert list(model.cross_b_from_a.parameters())
    else:
        assert not list(model.cross_b_from_a.parameters())


@pytest.mark.parametrize(
    ("enable_a_from_b", "enable_b_from_a"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_all_direction_combinations_support_cross_entropy_backward(
    enable_a_from_b: bool,
    enable_b_from_a: bool,
):
    model = _make_model(
        enable_a_from_b=enable_a_from_b,
        enable_b_from_a=enable_b_from_a,
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    logits = model(_make_inputs())["logits"]
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert torch.isfinite(loss)
    _assert_finite_participating_gradients(model)


@pytest.mark.parametrize(
    "bad_model_kwargs",
    [
        {"modalities": ["video", "can"]},
        {"modalities": ("video", "video")},
        {"modalities": ("video", "can"), "input_dims": {"video": 7}},
        {"modalities": ("video", "can"), "input_dims": {"video": 0, "can": 11}},
        {"num_classes": 1},
        {"d_model": 0},
        {"num_heads": 0},
        {"d_model": 7, "num_heads": 5},
        {"cross_layers": 0},
        {"memory_layers": 0},
        {"dropout": float("nan")},
        {"dropout": -0.1},
        {"dropout": 1.1},
        {"enable_a_from_b": 1},
        {"enable_b_from_a": 0},
        {"causal_attention": True},
    ],
)
def test_rejects_invalid_model_configuration(bad_model_kwargs: dict[str, object]):
    kwargs: dict[str, object] = {
        "modalities": ("video", "can"),
        "input_dims": {"video": 7, "can": 11},
        "num_classes": 3,
        "d_model": 10,
        "num_heads": 5,
        "cross_layers": 1,
        "memory_layers": 1,
        "dropout": 0.0,
        "enable_a_from_b": True,
        "enable_b_from_a": True,
    }
    kwargs.update(bad_model_kwargs)

    with pytest.raises((TypeError, ValueError)):
        DualModalMulT(**kwargs)


def test_rejects_non_mapping_inputs():
    model = _make_model()

    with pytest.raises((TypeError, ValueError)):
        model([])


def test_rejects_missing_or_extra_modality():
    model = _make_model()
    missing = _make_inputs()
    del missing["can"]
    with pytest.raises(ValueError):
        model(missing)

    extra = _make_inputs()
    extra["audio"] = _make_stream(3, 2, 13)
    with pytest.raises(ValueError):
        model(extra)


def test_rejects_non_mapping_stream_and_missing_fields():
    model = _make_model()
    non_mapping = _make_inputs()
    non_mapping["video"] = []  # type: ignore[assignment]
    with pytest.raises(ValueError):
        model(non_mapping)

    for field in ("x", "valid_mask", "time_s"):
        missing = _make_inputs()
        del missing["video"][field]
        with pytest.raises(ValueError):
            model(missing)


def test_rejects_extra_stream_fields():
    model = _make_model()
    inputs = _make_inputs()
    inputs["video"]["unexpected"] = torch.tensor(1.0)

    with pytest.raises(ValueError):
        model(inputs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda inputs: inputs["video"].update(
            x=inputs["video"]["x"][:, :, :6]
        ),
        lambda inputs: inputs["video"].update(
            valid_mask=inputs["video"]["valid_mask"].to(torch.int64)
        ),
        lambda inputs: inputs["video"].update(
            time_s=inputs["video"]["time_s"].to(torch.int64)
        ),
        lambda inputs: inputs["video"].update(
            valid_mask=inputs["video"]["valid_mask"][:, :-1]
        ),
        lambda inputs: inputs["video"].update(
            time_s=inputs["video"]["time_s"][:, :-1]
        ),
        lambda inputs: inputs["video"].update(
            x=inputs["video"]["x"][:, :, 0]
        ),
    ],
)
def test_rejects_wrong_stream_dtype_shape_or_feature_dimension(
    mutate: Callable[[dict[str, dict[str, torch.Tensor]]], None],
):
    inputs = _make_inputs()
    mutate(inputs)

    with pytest.raises(ValueError):
        _make_model()(inputs)


def test_rejects_mismatched_batch_sizes():
    inputs = _make_inputs()
    inputs["can"] = _make_stream(2, 3, 11)

    with pytest.raises(ValueError):
        _make_model()(inputs)


@pytest.mark.parametrize("field", ["x", "time_s"])
def test_rejects_nonfinite_values(field: str):
    inputs = _make_inputs()
    inputs["video"][field].reshape(-1)[0] = float("nan")

    with pytest.raises(ValueError):
        _make_model()(inputs)


def test_rejects_device_mismatch_before_tensor_values_are_read():
    inputs = _make_inputs()
    inputs["video"]["time_s"] = torch.empty(
        inputs["video"]["time_s"].shape,
        dtype=torch.float64,
        device="meta",
    )

    with pytest.raises(ValueError):
        _make_model()(inputs)


def test_rejects_zero_batch_and_zero_length_sequences():
    empty_batch = _make_inputs(
        masks=(
            torch.empty(0, 2, dtype=torch.bool),
            torch.empty(0, 3, dtype=torch.bool),
        )
    )
    with pytest.raises(ValueError):
        _make_model()(empty_batch)

    zero_length = _make_inputs(
        masks=(
            torch.empty(2, 0, dtype=torch.bool),
            torch.empty(2, 1, dtype=torch.bool),
        )
    )
    with pytest.raises(ValueError):
        _make_model()(zero_length)


def test_default_candidate_dimensions_are_configurable():
    model = DualModalMulT(
        modalities=("video", "can"),
        input_dims={"video": 7, "can": 11},
        num_classes=3,
    )

    assert model.d_model == 30
    assert model.num_heads == 5
    assert model.cross_layers == 5
    assert model.memory_layers == 5
    assert model.causal_attention is False
