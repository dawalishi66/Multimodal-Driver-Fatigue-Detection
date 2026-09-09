"""Extract frozen PANNs Cnn14_16k token embeddings for DCPT audio clips.

Per clip: decode WAV -> mean-channel mono -> 44.1k->16k resample (soxr) ->
five 2 s blocks (zero-pad only the trailing partial block, never stretch) ->
Cnn14_16k embedding per block -> one NPZ with the five standard arrays:
x[5,2048], time_s[5], valid_mask[5], support_s[5,2], observed_fraction[5].
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from panns_loader import load_cnn14_16k  # noqa: E402

SRC_SR = 44100
TARGET_SR = 16000
BLOCK_S = 2.0
N_BLOCKS = 5
BLOCK_SAMPLES = int(TARGET_SR * BLOCK_S)  # 32000
MIN_TOKEN_OBSERVED = 0.95


def read_stems(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        return [r["sample_id"] for r in rows]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_mono16k(entry_bytes: bytes) -> tuple[np.ndarray, float]:
    import soxr
    with wave.open(io.BytesIO(entry_bytes), "rb") as reader:
        channels = reader.getnchannels()
        sampwidth = reader.getsampwidth()
        framerate = reader.getframerate()
        if sampwidth != 2:
            raise ValueError(f"unsupported sample width {sampwidth}")
        raw = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1).astype(np.float32)
    else:
        raw = raw.astype(np.float32)
    audio = raw / 32768.0
    if framerate != TARGET_SR:
        audio = soxr.resample(audio, framerate, TARGET_SR).astype(np.float32)
    measured_s = float(audio.shape[0]) / TARGET_SR
    return audio, measured_s


def clip_to_arrays(audio: np.ndarray) -> dict[str, np.ndarray]:
    n = audio.shape[0]
    blocks = np.zeros((N_BLOCKS, BLOCK_SAMPLES), dtype=np.float32)
    observed = np.zeros(N_BLOCKS, dtype=np.float32)
    for k in range(N_BLOCKS):
        start = k * BLOCK_SAMPLES
        end = min(start + BLOCK_SAMPLES, n)
        if end > start:
            blocks[k, : end - start] = audio[start:end]
            observed[k] = (end - start) / BLOCK_SAMPLES
    valid = observed >= MIN_TOKEN_OBSERVED
    return {
        "blocks": blocks,
        "observed_fraction": observed,
        "valid_mask": valid,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, help="First_person_view_audio.zip")
    parser.add_argument("--stems", required=True, help="CSV/JSONL/txt providing sample_ids")
    parser.add_argument("--weights", required=True, help="Cnn14_16k_mAP=0.438.pth")
    parser.add_argument("--out-dir", required=True, help="processed/audio_features_v1")
    parser.add_argument("--index-out", required=True, help="feature index JSONL path")
    parser.add_argument("--summary-out", required=True, help="summary JSON path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="smoke limit (stems)")
    args = parser.parse_args(argv)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_cnn14_16k(args.weights, device)
    stems = read_stems(Path(args.stems))
    if args.limit:
        stems = stems[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index_out)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    index_lines: list[str] = []
    with zipfile.ZipFile(args.zip) as archive:
        for stem in stems:
            entry = f"First_person_view_audio/{stem}.wav"
            try:
                data = archive.read(entry)
                audio, measured = load_mono16k(data)
                prep = clip_to_arrays(audio)
                tensor = torch.from_numpy(prep["blocks"]).to(device)
                with torch.no_grad():
                    embedding = model(tensor)["embedding"].cpu().numpy().astype(np.float32)
                x = embedding
                time_s = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float64)
                support_s = np.array([[k * 2.0, k * 2.0 + 2.0] for k in range(N_BLOCKS)], dtype=np.float64)
                npz_path = out_dir / f"{stem}.npz"
                np.savez(
                    npz_path,
                    x=x,
                    time_s=time_s,
                    valid_mask=prep["valid_mask"],
                    support_s=support_s,
                    observed_fraction=prep["observed_fraction"],
                )
                digest = sha256_bytes(npz_path.read_bytes())
                rel = f"audio_features_v1/{stem}.npz"
                records.append({
                    "sample_id": stem, "modality": "audio",
                    "feature_version": "panns_cnn14_16k_v1", "path": rel,
                    "sha256": digest, "T": N_BLOCKS, "D": 2048,
                    "measured_duration_s": round(measured, 6), "status": "ok",
                })
                index_lines.append(json.dumps({
                    "sample_id": stem, "modality": "audio",
                    "feature_version": "panns_cnn14_16k_v1", "path": rel,
                    "sha256": digest, "T": N_BLOCKS, "D": 2048, "status": "ok",
                }))
            except Exception as exc:  # record, do not silently drop
                errors.append({"sample_id": stem, "error": f"{type(exc).__name__}: {exc}"})
                print(f"ERROR {stem}: {type(exc).__name__}: {exc}", flush=True)

    index_path.write_text("\n".join(index_lines) + ("\n" if index_lines else ""), encoding="utf-8")
    summary = {
        "feature_version": "panns_cnn14_16k_v1",
        "extractor_name": "panns_cnn14_16k",
        "clip_count": len(stems),
        "ok_count": len(records),
        "error_count": len(errors),
        "shape_T_D": {"T": N_BLOCKS, "D": 2048},
        "errors": errors,
        "device": str(device),
    }
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(f"ok={len(records)} errors={len(errors)} of {len(stems)}")
    print(f"index: {index_path}")
    print(f"summary: {args.summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())