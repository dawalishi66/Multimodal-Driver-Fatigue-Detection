# DCPT Distraction Audio Processing v1 (胡煦轩)

Status: raw audit, PANNs `[5, 2048]` feature extraction, the 6-class audio
single-modality baseline and the provisional video-alignment check are
implemented. The 6-class labels and 24/8/8 subject split remain provisional
until the project owners freeze them; no official DCPT result is claimed here.

## Confirmed naming schema (provisional until P04)

`First_person_view_audio.zip` census: 1080/1080 files match

    <task>_P<person>_<date>_<hour>_<minute>_<second>_<takeover>.wav

- `task` 01-09 -> fixed nine-class order (`DCPT_CLASSES`); this is the label.
- `person` P01-P40; `date` YYYYMMDD; HH/MM/SS = recording start; `takeover` in
  0.1 s units (pairing only, never a feature).
- The full stem is the cross-modal main identifier and is used as `sample_id`.

Parsing lives in exactly one place:
`src/driver_state/preprocessing/distraction_audio/naming.py`.

## QC semantics (`qc.py`)

Non-destructive audit with stdlib `wave` + numpy only. Never trims, re-samples,
re-channels, deletes silence, de-noises, stretches or interpolates.

| Check | Default | Result when violated |
| --- | --- | --- |
| Container decodes as WAV | - | `UNSUPPORTED_CONTAINER` (error row) |
| Samplerate | 44100 | `SAMPLE_RATE_DEVIATION` -> pending |
| Channels | 2 | `CHANNELS_NOT_STEREO` -> pending |
| Sample width | 2 (16-bit) | `SAMPLE_WIDTH_NOT_16BIT` -> pending |
| Non-empty | frames > 0 | `EMPTY_AUDIO` -> pending |
| Non-zero | any nonzero sample | `ZERO_AUDIO` -> pending (review, no auto-delete) |
| Clipping marker | extreme-sample ratio <= 1e-4 | `POSSIBLE_CLIPPING` flag only |
| Duration | nominal 10.0 s | see below |

Tail policy: shorter clips with shortfall <= 0.10 s -> `pass_with_flags`
(`MINOR_TAIL_SHORTFALL`, traceable); > 0.10 s -> `TAIL_SHORTFALL_EXCEEDS`
pending. Clips longer than nominal by > 0.05 s -> `LONGER_THAN_NOMINAL` pending
(no auto-trim). `raw_observed_fraction = min(measured, 10.0) / 10.0`.

## Metadata rows (`build_metadata.py`)

- One row per parseable DCPT clip, columns = `COMMON_METADATA_FIELDS`
  (21 fixed columns, schema 0.2.0).
- Nominal `[0, 10000)` ms, `window_index=0`. Measured facts live in the QC
  sidecar, never fabricated into the nominal window.
- Rows are `valid=false` + `error=NO_FEATURES_YET` until PANNs features exist.
  QC failures replace that error with their stable code. `valid_ratio=0` and
  feature columns are empty, which is what the public validator requires for
  feature-less rows.
- `extractor_name/version` = `dcpt_audio_qc_audit/0.1.0` (the audit tool that
  produced the row provenance); replaced by PANNs metadata in the next
  milestone.
- `split` is left empty until 李坤洋 freezes the 24/8/8 subject manifest; pass
  `--subject-splits <manifest.json>` to fill it (then the CSV can be validated).

## Outputs (written under `--output-dir`, kept out of git)

    metadata/audio_windows_10s_v1.csv
    qc/audio_qc_v1.json
    docs/audio_data_audit_v1.md

## Reproduce

```bash
python scripts/build_distraction_audio_metadata.py \
    --audio-source "<local>/First_person_view_audio.zip" \
    --output-dir "<local>/processed_audio" \
    --source-label "DCPT First_person_view_audio"

# after a frozen subject manifest exists:
python scripts/validate_distraction_audio_metadata.py \
    --metadata <output-dir>/metadata/audio_windows_10s_v1.csv \
    --feature-root <output-dir>
```

## Open items (owner-gated, not decided by code)

1. DCPT fixed subject split (24/8/8, seed 2026) - freeze by 李坤洋.
2. Final 6-class task mapping and label freeze with the video owner 陈星宇.
3. Official rerun and reporting only after both the split and labels above are frozen.

## Video-aligned 6-class fusion set (v1, 胡煦轩)

The distraction-video module (陈星宇) covers the six tasks that have upper-body
video (01/03/04/05/07/08, 714 clips) with provisional 6-class labels and a
24/8/8 seed-2026 subject split. The audio side mirrors it exactly so a later
fusion model can join by `sample_id`:

- `fusion_labels.py` holds the shared 6c task->label contract.
- `build_audio_6c_metadata.py` filters the 9c audit CSV to those six tasks and
  emits `metadata/audio_windows_10s_v1.csv` (video column order, valid rows once
  features exist).
- `validate_audio_6c.py` is a 6c-aware validator (the public validator is
  hard-coded to 9 classes).
- `check_fusion_pairs.py` asserts audio/video `sample_id`, label and split sets
  are identical (result: 714/714 common, 0 mismatches).

## Audio feature version v1 (PANNs Cnn14_16k)

Frozen `Cnn14_16k` (AudioSet-pretrained, `Cnn14_16k_mAP=0.438.pth`), input 16 kHz
mono waveform (mean-channel downmix, soxr resample from 44.1 kHz), five 2 s
blocks -> `[5, 2048]` embedding per clip. NPZ files carry the five standard
arrays; feature version `panns_cnn14_16k_v1`. The index version and the
weight-checkpoint provenance version are deliberately separate:
`panns_cnn14_16k_v1` is the feature contract, while
`cnn14_16k_mAP0.438_v1` identifies the extractor/weights.

`build_audio_6c_metadata.py` accepts only `status=ok` index rows. It rejects
duplicate or empty IDs, non-relative paths, malformed or non-finite NPZ arrays,
wrong dtype/shape, invalid time support or masks, SHA256/T/D mismatches, and
coverage below the configured threshold. Invalid features are not silently
converted to valid metadata. Stable error codes include:

- `FEATURE_INDEX_NOT_OK`
- `FEATURE_FILE_INVALID`
- `FEATURE_COVERAGE_BELOW_THRESHOLD`

Set `PANNS_VENDOR_ROOT` to the PANNs source root (or its `pytorch` directory)
when `panns_inference` is not installed. The loader accepts either location and
fails with the expected path when the vendor source cannot be found.

Local setup and the feature/training entry points:

```bash
python -m pip install -e ".[audio]"
python -m driver_state.preprocessing.distraction_audio.extract_audio_features \
  --zip "<local>/First_person_view_audio.zip" \
  --stems "<metadata-or-stem-list>" \
  --weights "<local>/Cnn14_16k_mAP=0.438.pth" \
  --out-dir "<local>/processed/audio_features_v1" \
  --index-out "<local>/processed/audio_feature_index_v1.jsonl" \
  --summary-out "<local>/processed/audio_feature_summary_v1.json"

python -m driver_state.models.unimodal.train_audio_baseline \
  --metadata "<local>/metadata/audio_windows_10s_v1.csv" \
  --feature-root "<local>/processed" \
  --label-scheme "<frozen-label-scheme.json>" \
  --artifact-dir "<local>/artifacts/audio_baseline_v1" \
  --summary-out "<local>/artifacts/audio_baseline_v1/summary.json" \
  --report-out "<local>/artifacts/audio_baseline_v1/report.md"
```

## Audio baseline v1 (fusion-aligned, 6c)

BiGRU hidden 192, dropout 0.25, mean+max+last pooling, AdamW 3e-4/5e-4, batch 32,
max_epochs 100, patience 15, val Macro-F1 selection, seeds 11/22/33. Test
Macro-F1 = 0.5682 +/- 0.0206 (video-only v4: 0.6910 +/- 0.0601, same cohort).
Per-clip predictions/metrics/checkpoints follow the video baseline file layout.
This is an honest single-modality result; fusion is a separate later step owned
by the fusion lead. The baseline excludes `valid=false` metadata rows before
training and records the excluded count and IDs in its config/summary. Its GRU
last pooling uses the final token with a true mask bit; an all-invalid mask is a
hard error.

`build_audio_evaluation.py` derives the video-v4-compatible `evaluation.json`
from the per-seed artifacts. It validates per-seed class arrays, supports and
confusion matrices before aggregating. Its per-class arrays contain one value
per class (6), not one value per seed, and the majority baseline is computed
from the first seed's test labels. On the current provisional cohort the
recomputed Macro-F1 is `0.5682041464 +/- 0.0205567034` with 144 test samples.
Test labels and predictions are used only for evaluation/reproduction, never
for tuning.
