# Distraction Video Baseline v4 Independent Recalculation

The existing v4 predictions were recomputed without retraining or changing any
model, threshold, sample or split decision.

| Seed | Test Macro-F1 | Balanced Accuracy | Accuracy |
|---:|---:|---:|---:|
| 11 | 0.621610 | 0.638331 | 0.631944 |
| 22 | 0.726315 | 0.733626 | 0.729167 |
| 33 | 0.724932 | 0.733928 | 0.729167 |

- Three-seed Macro-F1 mean: `0.690952`
- Three-seed sample standard deviation: `0.060056`
- Majority class from training labels: `2`
- Majority baseline accuracy: `0.180556`
- Majority baseline Macro-F1: `0.050980`

Stored metrics match the independent recalculation for all three seeds.

## Test Access Disclosure

The existing test set has already been accessed during v4 development. These
numbers are independent recomputations, not a new blind test. No result-driven
model or preprocessing change is permitted from this audit.

