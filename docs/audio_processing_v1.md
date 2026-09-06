# DCPT Distraction Audio Processing v1 (胡煦轩)

Status: raw audit + standard metadata tooling. PANNs `[5, 2048]` feature
extraction, the audio single-modality baseline and the P04 cross-modal pairing
with the video owner are **later milestones**; nothing here claims real-model
results or a frozen DCPT split.

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
2. Pre-feature valid-row policy (all `NO_FEATURES_YET` vs an exception).
3. P04: cross-modal main-id / common start verification with 陈星宇.
4. P07: PANNs feature extraction, feature-version freeze and `valid=true` flip.