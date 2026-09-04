"""CSV and feature-array contracts for one single-modality export."""

COMMON_METADATA_FIELDS = (
    "sample_id", "modality", "subject_id", "session_id", "split", "source_file",
    "window_index", "window_start_ms", "window_end_ms", "duration_ms",
    "label_class", "label_id", "valid", "valid_ratio", "mask", "feature_path",
    "feature_shape", "feature_dtype", "extractor_name", "extractor_version", "error",
)
FATIGUE_METADATA_FIELDS = (
    *COMMON_METADATA_FIELDS[:10],
    "label_start_ms", "label_end_ms", "kss_score",
    *COMMON_METADATA_FIELDS[10:],
)
DISTRACTION_METADATA_FIELDS = COMMON_METADATA_FIELDS
REQUIRED_FIELDS = {"fatigue": FATIGUE_METADATA_FIELDS, "distraction": DISTRACTION_METADATA_FIELDS}
FEATURE_DTYPES = {
    "x": "float32", "time_s": "float64", "valid_mask": "bool",
    "support_s": "float64", "observed_fraction": "float32",
}
