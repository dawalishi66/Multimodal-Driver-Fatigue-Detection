# UL-DD fatigue video baseline v1

This baseline closes the missing video-only comparator on the frozen
`fatigue_video_can_complete8_train_val_v1` cohort from PR #9.

## Fixed contract

- Model input: video only, `float32[6,96]`; CAN never enters the forward pass.
- Cohort: train 1,272 windows / 159 complete parents; validation 328 / 41.
- Labels: KSS `<4`, `[4,7)`, `>=7`, mapped to class IDs 0, 1, 2.
- Normalization: masked z-score fitted on valid train video tokens only.
- Model: 96-to-64 projection, one 64-unit GRU, dropout 0.2, 3-class logits.
- Seeds: 11, 22, 33. Selection and early stopping use validation parent
  Macro-F1; each parent probability is the mean of exactly eight window
  probability vectors.
- Status: train/validation development only. Test access is locked off.

The paired Dataset is intentionally reused to guarantee identical sample IDs,
parents, labels and split membership. CAN arrays are loaded only as part of the
already-audited paired Dataset contract and are neither normalized nor passed
to `VideoGruBaseline`.

## Run

From a clean committed worktree:

```powershell
python tools/fusion/fatigue_video_can/train.py `
  --dataset-root <UL-DD-root> `
  --pair-root <Paired_Video_CAN_v1-root> `
  --config configs/baselines/fatigue_video/gru_v1_train_val.json `
  --device auto
```

The external run package is written below
`<pair-root>/baselines/fatigue_video/video_gru_v1/runs/`. It includes the
resolved config, environment, train-only preprocessing state, per-seed
checkpoint, logs, validation predictions, parent predictions, metrics,
confusion matrices, coverage report, hashes and reload verification. Weights
and full predictions must remain outside Git.

For a one-seed smoke before the full run, append `--seeds 11`. Do not add a
test manifest, test metric or test-dependent threshold/model choice.

## Locked C/H/P feature handoff

`tools/baselines/fatigue_video/test_features.py` prepares the public test
feature package after the team lead provides a written unlock. It does not
extract raw video itself: the frozen provider-side extractor first emits the
five-array 30-second NPZ files, then this tool verifies `[6,96]`, converts
session-relative times to sample-relative 0–30 seconds, records SHA-256, and
revalidates the completed package.

The request CSV columns are exactly:

```text
sample_id,subject_id,session_id,window_start_ms,window_end_ms,source_feature,source_time_reference
```

It must cover only and all subjects C/H/P and must contain no KSS, label,
probability, prediction or metric field. The unlock JSON must contain:

```json
{
  "authorized_by": "<team lead>",
  "issued_at": "<timestamp>",
  "scope": "test_feature_generation_and_validation_only",
  "allow_model_evaluation": false,
  "allow_threshold_or_model_changes": false
}
```

After authorization, export and independently validate with:

```powershell
python tools/baselines/fatigue_video/test_features.py export `
  --unlock <written-unlock.json> `
  --request <test-feature-request.csv> `
  --source-root <provider-feature-root> `
  --output-root <new-test-handoff-directory>

python tools/baselines/fatigue_video/test_features.py validate `
  --unlock <written-unlock.json> `
  --output-root <new-test-handoff-directory>
```

Both commands are feature-only. The report fixes `model_loaded=false`,
`predictions_generated=false`, `metrics_evaluated=false`, and
`thresholds_or_models_changed=false`. This task did not run these commands on
real C/H/P video features because no written unlock was supplied.
