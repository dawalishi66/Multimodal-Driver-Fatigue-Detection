# Video (DCPT distraction) artifacts - v1 index

Large artifacts remain in controlled external storage and are not committed.

| artifact_id | type | version | count/size | sha256 | note |
| --- | --- | --- | --- | --- | --- |
| r3d18_kinetics400_weights | model weights | torchvision R3D-18 Kinetics-400 | 1 file | B3B3357EAD25631EC9C57362FF2128A92D0427E01E2CD184951A44380C3F2E9D | official torchvision weights |
| video_features_v1 | feature NPZ cache | r3d18_kinetics400_v1 | 714 files, [10,512], 15735132 bytes | per-file in feature_index_v1.jsonl | five standard arrays |
| video_metadata_6c_v1 | metadata CSV | dcpt_video_6c_v1 | 714 rows | E171DC56D43AB2700E64BBF8D23736C9FB2919914B27BED1F5EAE05A5C100E67 | formal six-class view; original artifact not overwritten |
| video_baseline_v4 | model+metrics | v4, seeds 11/22/33 | 3 checkpoints | per-run best.pt | recomputed test Macro-F1 0.6910 +/- 0.0601; test already accessed |
