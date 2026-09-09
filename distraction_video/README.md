# Distraction Video Module

Owner: Chen Xingyu
Task: DCPT upper-body video preprocessing and video single-modality baseline.

## Scope

- Input: 714 DCPT upper-body video clips (tasks 01/03/04/05/07/08).
- Feature route: frozen R3D-18 with `KINETICS400_V1` weights.
- Output per clip: `[10, 512]` float32 feature sequence with masks and time metadata.
- No fusion or MulT work is implemented in this module.

## Current Artifacts

- `tools/` scripts that build the sample manifest and extract features.
- `configs/` local and example feature extraction configuration.
- `metadata/` standard `video_windows_10s_v1.csv`, feature sidecar, provisional manifest, label scheme, subject split, and training samples.
- `reports/` extraction summary, low-coverage sample list, validation report, and the selected v4 baseline report.
- `processed/` feature index and cached NPZ features with exactly the five standard arrays.

`python tools/validate_distraction_video_metadata.py` checks the schema 0.2.0
structure of the metadata CSV and NPZ files.

Raw extracted videos and full NPZ features stay outside this repository and are
referenced through the feature index. The local config contains machine paths
and is ignored; copy it from the example file when running in a new location.

## Selected Baseline

The current selected video-only baseline is `v4`: bidirectional GRU, hidden
size 192, dropout 0.25, mean+max+last pooling, and AdamW weight decay 0.0005.
The three-seed test Macro-F1 is 0.6910 +/- 0.0601.

## Planned Layout

The baseline model, training config, checkpoints index, and experiment results
will be added under:

```text
distraction_video/
  artifacts/baseline_v1/
  runs/<run_id>/
```
