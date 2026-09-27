# DCPT Distraction Video Data Audit v1

Status: automated audit complete; split freeze and P04 human review pending.

## Scope

- Formal task: DCPT six-class subset tasks 01/03/04/05/07/08
- Labels: 0 No task, 1 Playing game, 2 Messaging, 3 Phone call, 4 Reading, 5 Eating
- Candidates: 714 clips from 40 subjects
- Original tasks 02/06/09 exist but are outside this version.

## Feature and Metadata Verification

- Feature files checked: 714
- Feature shape and dtype: `float32[10,512]`
- Required NPZ arrays: 5/5 for every checked file
- NaN/Inf findings: 0 errors
- SHA-256 mismatches: 0
- Valid rows: 692
- Invalid rows retained with error: 22
- Feature coverage range: 0.900 to 1.000

## Split Candidate

The audit uses the existing `dcpt_subject_24_8_8_seed2026_v1_provisional` subject map without
rerandomization. It is exported as `dcpt_subject_24_8_8_seed2026_v1_freeze_candidate` with
`candidate_awaiting_owner_signoff`. 李坤洋 must confirm the exact 24/8/8 list
before the candidate can be called frozen.

Split summary:

```json
{
  "train": {
    "subjects": 24,
    "samples": 419,
    "class_counts": {
      "0": 69,
      "1": 70,
      "2": 73,
      "3": 67,
      "4": 71,
      "5": 69
    },
    "valid_true": 411,
    "valid_false": 8
  },
  "val": {
    "subjects": 8,
    "samples": 151,
    "class_counts": {
      "0": 25,
      "1": 29,
      "2": 25,
      "3": 23,
      "4": 26,
      "5": 23
    },
    "valid_true": 145,
    "valid_false": 6
  },
  "test": {
    "subjects": 8,
    "samples": 144,
    "class_counts": {
      "0": 23,
      "1": 24,
      "2": 26,
      "3": 23,
      "4": 24,
      "5": 24
    },
    "valid_true": 136,
    "valid_false": 8
  }
}
```

## Quality and Timing Evidence

- Feature extraction sidecar rows: 714
- Effective frame-rate min/median/max: 26.742536 / 30.105351 / 30.122408
- `observed_fraction` and `valid_mask` are retained per token.
- Frame selection and segment support follow actual decoded timestamps in the extractor.
- Raw video re-decode: not executed because raw videos are absent from the current environment.

## Open Gates

1. 李坤洋 sign-off for the exact split/cohort candidate.
2. 胡煦轩 and 陈星宇 manual P04 playback review and signatures.
3. Raw-frame replay when raw videos are available again.
4. Test access history remains disclosed; no tuning is allowed from the existing test scores.

## Cross-Modal ID Check

The automated audio/video comparison is recorded in
`results/distraction_video/video_audio_pair_check_v1.json`. It checks IDs,
subject/session/split/label and feature-index presence, but cannot replace P04
human synchronization evidence.

