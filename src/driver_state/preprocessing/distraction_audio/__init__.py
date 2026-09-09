"""Owner: 胡煦轩. DCPT distraction-audio raw audit and metadata export.

Implemented so far (lightweight, numpy + stdlib only):

* ``naming``: DCPT filename -> subject/task/label/main-id mapping;
* ``qc``: WAV format + duration + non-destructive quality markers;
* ``build_metadata``: metadata CSV + QC sidecar + audit markdown (CLI).

PANNs feature extraction ([5, 2048] NPZ), the audio single-modality baseline and
the P04 cross-modal pairing with the video owner are later milestones.
"""