# DCPT Video Feature Card v1 (Formal Six-Class Subset)

- Owner: 陈星宇
- Feature version: `r3d18_kinetics400_v1`
- Input view: DCPT upper-body video
- Extractor: frozen torchvision R3D-18, Kinetics-400 weights
- Weight SHA-256: `B3B3357EAD25631EC9C57362FF2128A92D0427E01E2CD184951A44380C3F2E9D`
- Per-sample output: `x float32[10, 512]`
- Time arrays: `time_s float64[10]`, `support_s float64[10,2]`
- Validity arrays: `valid_mask bool[10]`, `observed_fraction float32[10]`
- Sampling: ten 1-second feature segments; frame selection uses actual video timestamps, not a hard-coded 30 fps assumption.
- NPZ contains exactly the five standard arrays and no pickle objects.

## Audit Totals

- 714 candidate clips
- 714 feature files checked
- 692 rows marked valid
- 22 rows retained as invalid with explicit error codes
- Total feature bytes: 15735132
- Feature coverage range: 0.900 to 1.000
- Sidecar effective frame-rate range: 26.742536 to 30.122408, median 30.105351

The current environment does not contain the raw upper-body videos, so a new
raw-frame re-decode was not executed. The existing extraction sidecar and the
timestamp-aware extractor are the available evidence; this limitation is not a
PASS claim for raw-frame replay.

