"""Extract frozen R3D-18 Kinetics-400 features for DCPT upper-body videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.video import R3D_18_Weights
from torchvision.transforms.functional import center_crop, convert_image_dtype, normalize, resize


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_model(config: dict[str, Any], device: torch.device):
    state_dict = torch.load(config["weights_path"], map_location="cpu", weights_only=False)
    model = torchvision.models.video.r3d_18(weights=None)
    model.load_state_dict(state_dict)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def decode_frames(video_path: Path, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        times: list[float] = []
        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format="rgb24")
            img = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)
            img = resize(
                img,
                list(target_size),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=False,
            )
            frames.append(img[0].permute(1, 2, 0).numpy().copy())
            times.append(float(frame.time))
    finally:
        container.close()
    if not frames:
        raise RuntimeError("no decoded video frames")
    return np.asarray(times, dtype=np.float64), np.stack(frames, axis=0)


def select_segment_frames(
    times: np.ndarray,
    segment_index: int,
    frames_per_second: int,
) -> tuple[np.ndarray, float, int]:
    lo = float(segment_index)
    hi = float(segment_index + 1)
    idx = np.where((times >= lo) & (times < hi))[0]
    if idx.size == 0:
        nearest = int(np.argmin(np.abs(times - (lo + 0.5))))
        selected = np.full(frames_per_second, nearest, dtype=np.int64)
        return selected, 0.0, 0

    dt = float(np.median(np.diff(times[idx]))) if idx.size > 1 else 1.0 / 30.0
    observed_fraction = min(1.0, float(idx.size * dt))
    if idx.size >= frames_per_second:
        targets = lo + (np.arange(frames_per_second) + 0.5) / frames_per_second
        positions = np.searchsorted(times[idx], targets, side="left")
        positions = np.clip(positions, 0, idx.size - 1)
        selected = idx[positions]
    else:
        selected = idx[np.linspace(0, idx.size - 1, frames_per_second).astype(np.int64)]
    return selected, observed_fraction, int(idx.size)


def preprocess_segment(frames: np.ndarray, transform_metadata: dict[str, Any]) -> torch.Tensor:
    vid = torch.from_numpy(frames).permute(0, 3, 1, 2)
    vid = resize(
        vid,
        list(transform_metadata["resize_size"]),
        interpolation=transform_metadata["interpolation"],
        antialias=False,
    )
    vid = center_crop(vid, list(transform_metadata["crop_size"]))
    vid = convert_image_dtype(vid, torch.float)
    vid = normalize(vid, transform_metadata["mean"], transform_metadata["std"])
    return vid.permute(1, 0, 2, 3)


def process_sample(
    video_path: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    transform_metadata: dict[str, Any],
) -> dict[str, Any]:
    target_size = tuple(transform_metadata["resize_size"])
    times, frames = decode_frames(video_path, target_size)

    segments: list[torch.Tensor] = []
    observed = np.empty(config["segments"], dtype=np.float32)
    valid = np.zeros(config["segments"], dtype=bool)
    frame_counts = np.zeros(config["segments"], dtype=np.int64)

    for segment_index in range(config["segments"]):
        selected, observed_fraction, count = select_segment_frames(
            times,
            segment_index,
            config["frames_per_second"],
        )
        observed[segment_index] = observed_fraction
        frame_counts[segment_index] = count
        valid[segment_index] = observed_fraction >= config["min_valid_raw_fraction"]
        segments.append(preprocess_segment(frames[selected], transform_metadata))

    batch = torch.stack(segments, dim=0).to(device)
    with torch.no_grad():
        features = model(batch)
    features = features.detach().cpu().numpy().astype(np.float32)
    features[~valid] = 0.0

    time_s = (np.arange(config["segments"]) + 0.5).astype(np.float64)
    support_s = np.stack(
        [
            np.arange(config["segments"], dtype=np.float64),
            np.arange(1, config["segments"] + 1, dtype=np.float64),
        ],
        axis=1,
    )
    return {
        "x": features,
        "time_s": time_s,
        "valid_mask": valid,
        "support_s": support_s,
        "observed_fraction": observed,
        "raw_frame_count": frame_counts,
        "decoded_frame_count": int(times.size),
        "decoded_start_s": float(times[0]),
        "decoded_end_s": float(times[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distraction_video_feature_v1.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest_path"])
    output_dir = Path(config["output_dir"])
    index_path = Path(config["index_path"])
    log_path = Path(config["log_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        manifest = manifest[: args.limit]

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    transform = R3D_18_Weights.KINETICS400_V1.transforms()
    transform_metadata = {
        "crop_size": list(transform.crop_size),
        "resize_size": list(transform.resize_size),
        "mean": list(transform.mean),
        "std": list(transform.std),
        "interpolation": transform.interpolation,
    }

    extracted_dir = Path(config["extracted_dir"])
    existing = set()
    if args.resume and index_path.exists():
        existing = {
            json.loads(line)["sample_id"]
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    for sample_index, sample in enumerate(manifest, start=1):
        sample_id = sample["sample_id"]
        if args.resume and sample_id in existing:
            continue
        source_file = sample["source_refs"]["video_file"]
        video_path = extracted_dir / source_file
        output_path = output_dir / f"{sample_id}.npz"
        try:
            result = process_sample(
                video_path,
                model,
                config,
                device,
                transform_metadata,
            )
            np.savez(str(output_path), **result)
            index_record = {
                "sample_id": sample_id,
                "modality": "video",
                "feature_version": config["feature_version"],
                "path": str(output_path.relative_to(config["data_root"])),
                "sha256": sha256_file(output_path),
                "T": int(result["x"].shape[0]),
                "D": int(result["x"].shape[1]),
                "status": "ok",
            }
            with index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(index_record, ensure_ascii=False) + "\n")
            log_record = {
                "sample_id": sample_id,
                "status": "ok",
                "valid_tokens": int(result["valid_mask"].sum()),
                "decoded_frame_count": result["decoded_frame_count"],
                "decoded_start_s": result["decoded_start_s"],
                "decoded_end_s": result["decoded_end_s"],
            }
        except Exception as exc:  # keep one bad file from stopping the batch
            log_record = {
                "sample_id": sample_id,
                "status": "error",
                "error": type(exc).__name__,
                "detail": str(exc)[:500],
            }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        if sample_index % 10 == 0 or sample_index == len(manifest):
            print(f"processed {sample_index}/{len(manifest)}", flush=True)


if __name__ == "__main__":
    main()
