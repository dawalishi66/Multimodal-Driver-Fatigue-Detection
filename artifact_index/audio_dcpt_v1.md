# Audio (DCPT distraction) artifacts - v1 index

Entries live in the local controlled storage deliverable `distraction_audio/`
(next to the DCPT data; not in this repository). Verify SHA-256 after download.

| artifact_id | type | version | count/size | sha256 | note |
| --- | --- | --- | --- | --- | --- |
| panns_cnn14_16k_weights | model weights | Cnn14_16k_mAP=0.438 | 358,668,570 B | E2EE543A27919542C2EA03EABAA70B24DCD4E6C8E05621DE6B67A94E4C5058E6 | official PANNs (16k) |
| audio_features_v1 | feature NPZ cache | panns_cnn14_16k_v1 | 714 files, [5,2048] | per-file in audio_feature_index_v1.jsonl | five standard arrays |
| audio_metadata_6c_v1 | metadata CSV | dcpt_audio_6c_v1 | 714 rows | see metadata sidecar | video-aligned |
| audio_baseline_v1 | model+metrics | v1 (seeds 11/22/33) | 3 checkpoints | per-run metrics.json | test Macro-F1 0.5682+/-0.0206 |