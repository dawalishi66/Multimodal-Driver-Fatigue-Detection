# Fusion Models v1

## 1. Scope and Ownership

This document describes the two model deliveries owned by 饶棋涛. The implementation is the current repository code, not a promise of a completed real-data experiment.

Responsibilities:

- Simple Fusion;
- Dual-modal MulT;
- model configuration templates;
- structure, input/output, shape-flow, and mask documentation;
- mask and forward/backward tests;
- necessary direction-ablation switches.

This ownership does not include raw preprocessing, `Dataset`/collate, the public training loop, formal metrics, formal experiment scheduling, or changing labels and splits.

The templates in `configs/simple_fusion_v1.example.json` and `configs/dual_modal_mult_v1.example.json` are examples. Each task profile is resolved independently; one model instance must not run fatigue and distraction together.

## 2. Fixed Task Contract

| Task | Dataset | Modalities | Classes |
|---|---|---|---:|
| fatigue | UL-DD | video + can | 3 |
| distraction | DCPT | video + audio | 9 |

Both models consume a mapping whose selected streams contain:

```text
x          [B, T, D]
valid_mask [B, T]
time_s     [B, T]
```

`valid_mask=True` means that the token is valid. The model output is exactly:

```python
{"logits": tensor[B, C]}
```

The model returns raw logits, not softmax probabilities, loss values, hidden states, or attention weights. The caller supplies actual `input_dims` after resolving the verified feature manifest; `Dv`, `Dc`, and `Da` are not known model-template constants.

## 3. SimpleFusion

The current `SimpleFusion` constructor is:

```python
SimpleFusion(
    modalities,
    input_dims,
    num_classes,
    projection_dim,
    classifier_hidden_dim,
    dropout,
)
```

Its actual flow is:

```text
Modality A [B, Ta, Da]
  -> independent Linear(Da, projection_dim)
  -> GELU + dropout
  -> mask-aware mean pooling
  -> [B, H]

Modality B [B, Tb, Db]
  -> independent Linear(Db, projection_dim)
  -> GELU + dropout
  -> mask-aware mean pooling
  -> [B, H]

concat [B, 2H]
  -> Linear(2H, classifier_hidden_dim)
  -> GELU + dropout
  -> Linear(classifier_hidden_dim, C)
  -> logits [B, C]
```

`Ta` and `Tb` may differ, and internal invalid tokens are excluded from the mean. `time_s` is checked as an interface field but does not enter prediction. No real feature dimension is hard-coded.

The template values `projection_dim=5`, `classifier_hidden_dim=4`, and `dropout=0.2` mirror the current synthetic test setup only. They are marked `first_version_example`; v0.2 does not establish them as mandatory or optimal values.

## 4. DualModalMulT

The current `DualModalMulT` constructor is:

```python
DualModalMulT(
    modalities,
    input_dims,
    num_classes,
    d_model=30,
    num_heads=5,
    cross_layers=5,
    memory_layers=5,
    dropout=0.0,
    enable_a_from_b=True,
    enable_b_from_a=True,
    causal_attention=False,
)
```

For ordered modalities `(A, B)`, the shape flow is:

```text
A [B, Ta, Da]
  -> Linear(Da, d), bias=False, sequence position
  -> [B, Ta, d]

B [B, Tb, Db]
  -> Linear(Db, d), bias=False, sequence position
  -> [B, Tb, d]

A <- B: Q=A, K/V=B
  -> [B, Ta, d]

B <- A: Q=B, K/V=A
  -> [B, Tb, d]

A target stream -> target-specific memory self-attention
  -> [B, Ta, d]

B target stream -> target-specific memory self-attention
  -> [B, Tb, d]

last-valid(A) -> [B, d]
last-valid(B) -> [B, d]

concat -> [B, 2d]
  -> residual MLP -> [B, 2d]
  -> classifier -> [B, C] logits
```

Each attention/FFN block is pre-LN, uses `MultiheadAttention(batch_first=True)`, an FFN `d -> 4d -> d` with ReLU, residual paths, dropout, and final stack normalization. Causal attention is disabled and `causal_attention=True` is rejected. This is a true two-modal version: there is no fake text or third modality.

## 5. Relation to Original MulT

The architectural source is Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences* (ACL 2019), with the authors' reference implementation at [yaohungt/Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer). The repository describes directional pairwise crossmodal transformers followed by sequence models; its model defines independent projections, directional crossmodal streams, target-specific memories, and a residual prediction head.

This repository says `architecture adapted from`, not `copied from`: the current code uses PyTorch `MultiheadAttention`, explicit repository masks, and the v0.2 batch contract. The necessary retained structure is:

- independent modality projection;
- directional crossmodal attention;
- target-specific memory self-attention;
- sequence positional encoding;
- residual prediction head.

The required adaptation is:

- 3 modalities -> 2 modalities;
- 6 cross directions -> 2 directions;
- memory input `2d` -> `d`;
- final concat `6d` -> `2d`;
- regression/emotion output -> classification logits;
- old custom attention -> PyTorch `MultiheadAttention`;
- explicit `valid_mask` support;
- last padded token -> last valid token.

These are task and interface adaptations, not separate research innovations.

## 6. Mask Semantics

The project uses `True = valid`. PyTorch `key_padding_mask` uses `True = ignore`, therefore every attention call receives:

```python
key_padding_mask = ~valid_mask
```

The implementation:

- zeros invalid values before projection;
- masks source keys for both cross directions and each memory stream;
- zeros invalid target states again after attention, residual, FFN, and normalization;
- preserves internal invalid slots instead of compacting sequences;
- rejects a batch row with an all-invalid modality before the first attention call;
- does not use `x == 0` as an invalid marker.

For example, with:

```text
[True, False, True, False]
```

the last valid slot is index `2`, not `valid_count - 1`. Tail padding may be appended, but it does not change the original slot positions.

## 7. Positional Encoding

`DualModalMulT` uses deterministic sequence sinusoidal positions. Positions are based on original sequence slots, not on `time_s`; internal invalid slots do not cause later valid tokens to be renumbered, and tail padding does not change existing token positions. The projected stream is scaled by `sqrt(d_model)`, invalid slots are cleared after adding position, and the encoding follows the target tensor device and dtype, including legal odd `d_model` values.

`time_s` remains required, floating-point, finite, and shape-checked because it is part of the shared interface. The first version does not use real-time encoding, and causal attention is disabled. This documentation does not claim real temporal modeling beyond sequence position.

## 8. Ablation

The two constructor switches are `enable_a_from_b` and `enable_b_from_a`, where A and B are the ordered entries of `modalities`.

| `enable_a_from_b` | `enable_b_from_a` | Meaning |
|---:|---:|---|
| `true` | `true` | complete bidirectional baseline MulT |
| `true` | `false` | A <- B only |
| `false` | `true` | B <- A only |
| `false` | `false` | no-cross control |

When a direction is disabled, its cross blocks are not instantiated or computed. The target stream still enters its own memory path and the final representation remains `[B, 2d]`. The no-cross control is not called a complete MulT model.

## 9. G1 Verification

The current synthetic CPU G1 evidence is:

- `tests/test_simple_fusion.py`: 34 collected tests;
- `tests/test_dual_modal_mult.py`: 55 passed;
- full repository suite: 206 passed.

The tests cover task output shapes, `Ta != Tb`, heterogeneous valid masks, internal invalid tokens, single-token inputs, invalid-value and tail-padding invariance, MulT `time_s` independence, last-valid selection, all-invalid rejection, input contract failures, finite forward results, CrossEntropy backward, finite participating gradients, direction ablations, and CPU execution.

These results establish synthetic G1 interface behavior only. They do not establish real feature quality, synchronization, cohort validity, or model superiority.

## 10. G2 Integration Requirements

Real-data G2 requires, without implementing it in this delivery:

- verified feature dimensions `Dv`, `Dc`, and `Da`;
- feature version and extractor/preprocessing record;
- paired sample IDs and the fixed public cohort;
- validated `valid_mask` values;
- validated time mapping and support information;
- a small real batch for read, forward, loss/gradient, save, and reload checks.

These missing items do not invalidate synthetic G1. Until they are verified, the project must not claim that real-data fusion is complete.

## 11. Known Limits

- No real-data G2 integration has been completed.
- No formal training has been completed.
- No formal test evaluation has been completed.
- The results do not show that MulT outperforms SimpleFusion.
- No real feature route has been accepted by the data-quality process.
- Pruning, distillation, and quantization are not implemented.
- PyTorch dependency declarations and public CI are repository-level integration matters; this delivery does not modify them.

Formal labels, splits, preprocessing, training, metrics, and experiment scheduling remain governed by the v0.2 specification and their respective owners.
