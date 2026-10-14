"""Owner: 胡煦轩. DCPT distraction-audio preprocessing, features and baseline.

* ``naming``: DCPT filename -> subject/task/label/main-id/session mapping;
* ``qc``: WAV format + duration + non-destructive quality markers;
* ``build_metadata``: nine-class raw-audit metadata CSV + QC sidecar + audit md;
* ``fusion_labels``: the shared ``dcpt_video_6c_v1`` label/split contract;
* ``build_audio_6c_metadata``: video-aligned six-class metadata CSV (CLI);
* ``validate_audio_6c``: six-class-aware structural + feature validator;
* ``check_fusion_pairs``: audio/video ``sample_id``/label/split/session parity;
* ``extract_audio_features``: frozen PANNs Cnn14_16k ``[5, 2048]`` NPZ cache;
* ``build_audio_evaluation``: video-v4-compatible evaluation aggregate.

The single-modality baseline model lives in
``driver_state.models.unimodal.train_audio_baseline``.

The six-class label scheme and the 24/8/8 subject split are provisional mirrors
of the distraction-video module until the project lead freezes them; the P04
cross-modal synchronisation check with the video owner is still open.
"""
