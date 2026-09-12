# DCPT Distraction Video Processing v1 (陈星宇)

Status: standard 6-class video metadata, feature index, validation and
single-modality baseline. This document describes the module structure and the
provisional interface used by the video branch.

## Module layout

```text
src/driver_state/preprocessing/distraction_video/  # reusable implementation
scripts/                                            # thin CLI wrappers
tests/test_distraction_video_*                      # synthetic tests
distraction_video/                                  # metadata, configs, reports
artifact_index/video_dcpt_v1.md                     # external artifact index
```

## Naming and labels

```text
<task>_P<person>_<date>_<hour>_<minute>_<second>_<takeover>.mp4
```

`sample_id` is the full file stem. The video subset uses tasks
`01/03/04/05/07/08`, mapped to:

```text
0 No task
1 Playing game
2 Messaging
3 Phone call
4 Reading
5 Eating
```

The provisional label scheme is `dcpt_video_6c_v1`; the subject split is
`dcpt_subject_24_8_8_seed2026_v1_provisional` (24/8/8, seed 2026).

## Metadata and validation

The standard CSV stores one nominal `[0,10000)` millisecond clip per row.
Feature NPZ files contain exactly five arrays:

```text
x[10,512] float32, time_s[10] float64, valid_mask[10] bool,
support_s[10,2] float64, observed_fraction[10] float32
```

The six-class validator checks metadata structure, label/split alignment and
the NPZ arrays. Feature caches, raw video and model weights stay outside git.

## Video baseline v4

Bidirectional GRU, hidden 192, dropout 0.25, mean+max+last pooling, AdamW
`3e-4/5e-4`, batch 32, patience 15, seeds `11/22/33`. The three-seed test
Macro-F1 is `0.6910 +/- 0.0601` on the provisional 714-clip cohort.
