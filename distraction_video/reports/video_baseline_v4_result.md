# Distraction Video Baseline v4 Result

## Changes from v2

- Pooling changed from mean + last token to mean + max + last token.
- Weight decay increased from 0.0001 to 0.0005.
- Hidden size 192, dropout 0.25, and bidirectional GRU remain unchanged.

The v4 configuration was selected on validation before running the final test.

## Test Results

| Seed | Best val Macro-F1 | Test Macro-F1 | Test balanced accuracy | Test accuracy |
|---|---:|---:|---:|---:|
| 11 | 0.7598 | 0.6216 | 0.6383 | 0.6319 |
| 22 | 0.7505 | 0.7263 | 0.7336 | 0.7292 |
| 33 | 0.7389 | 0.7249 | 0.7339 | 0.7292 |

Mean test Macro-F1: 0.6910 +/- 0.0601.

Compared with v2: test Macro-F1 improves from 0.6853 to 0.6910, and balanced
accuracy improves from 0.6910 to 0.7020. The seed variance is larger.

## Per-class mean F1

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| No task | 23 | 0.609 | 0.797 | 0.684 |
| Playing game | 24 | 0.861 | 0.500 | 0.631 |
| Messaging | 26 | 0.566 | 0.372 | 0.436 |
| Phone call | 23 | 0.819 | 0.696 | 0.743 |
| Reading | 24 | 0.986 | 0.917 | 0.950 |
| Eating | 24 | 0.568 | 0.931 | 0.702 |

## Per-subject Macro-F1 (mean over seeds)

| Subject | Samples | Macro-F1 |
|---|---:|---:|
| P07 | 20 | 0.573 |
| P08 | 18 | 0.573 |
| P15 | 17 | 0.594 |
| P21 | 18 | 0.833 |
| P27 | 18 | 0.561 |
| P32 | 17 | 0.780 |
| P33 | 18 | 0.562 |
| P38 | 18 | 0.773 |

## Caveats

- Provisional 6-class labels and 24/8/8 subject split.
- Official results require the shared manifest and split from the project lead.
