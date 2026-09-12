"""Owner: 陈星宇. DCPT distraction-video metadata, QC and interface helpers.

Implemented so far (numpy + stdlib only):

* ``naming``: DCPT filename -> subject/task/label/session mapping;
* ``fusion_labels``: the provisional six-class video contract;
* ``build_metadata``: standard video CSV from the local manifest;
* ``validate_video_6c``: six-class-aware structural and feature validation;
* ``check_pairs``: sample/label/split/session alignment with the audio module.

Feature extraction, training and evaluation remain in ``distraction_video/tools``.
"""
