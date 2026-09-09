"""Rewrite DCPT video NPZ files to the five-array public interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STANDARD_ARRAYS = [
    "x",
    "time_s",
    "valid_mask",
    "support_s",
    "observed_fraction",
]


def normalize(feature_dir: Path, sidecar_path: Path) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_rows = []
    for npz_path in sorted(feature_dir.glob("*.npz")):
        data = np.load(npz_path)
        sidecar = {
            "sample_id": npz_path.stem,
            "feature_version": "r3d18_kinetics400_v1",
        }
        for key in (
            "raw_frame_count",
            "decoded_frame_count",
            "decoded_start_s",
            "decoded_end_s",
        ):
            if key in data.files:
                value = data[key]
                if value.ndim == 0:
                    sidecar[key] = value.item()
                else:
                    sidecar[key] = value.tolist()
        output = {key: data[key] for key in STANDARD_ARRAYS}
        np.savez(npz_path, **output)
        sidecar_rows.append(sidecar)

    with sidecar_path.open("w", encoding="utf-8") as handle:
        for row in sidecar_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"normalized {len(sidecar_rows)} files")
    print(f"sidecar {sidecar_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--sidecar-path", required=True)
    args = parser.parse_args()
    normalize(Path(args.feature_dir), Path(args.sidecar_path))


if __name__ == "__main__":
    main()
