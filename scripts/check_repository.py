"""Lightweight checks for repository content, not a complete secret scanner."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = {"data", "raw", "processed", "artifacts", "runs", "checkpoints", "outputs", ".venv", ".secrets"}
FORBIDDEN_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".hdf5",
    ".mp4", ".avi", ".mov", ".wav", ".mp3", ".zip", ".7z", ".pkl", ".pickle", ".pem", ".key",
}
REQUIRED = (
    "README.md", "CONTRIBUTING.md", "pyproject.toml", "requirements.txt",
    "docs/EXPERIMENT_RULES.md", "docs/INTERFACES.md", "docs/TEAM_AND_AI.md",
    "configs/fatigue_uldd.json", "configs/distraction_dcpt.json", "configs/paths.example.json",
    ".github/pull_request_template.md", ".github/workflows/ci.yml",
)
ABSOLUTE_LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"']+|/(?:home|Users)/[^\s\"']+")
TOKEN = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{32,}|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----)")


def main() -> int:
    errors: list[str] = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append(f"missing required file: {name}")
    command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT)]
    try:
        result = subprocess.run(command + ["ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        print("FAIL: Git file inventory is unavailable")
        return 1
    names = sorted(set(result.stdout.decode("utf-8").rstrip("\0").split("\0")) - {""})
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        parts = Path(name).parts
        if (parts[0] in FORBIDDEN_ROOTS or path.suffix.lower() in FORBIDDEN_SUFFIXES
                or path.name == ".env" or path.name.startswith(".env.") or ".local." in path.name):
            errors.append(f"forbidden repository content: {name}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file exceeds the 10 MiB repository policy: {name}")
            continue
        if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".toml", ".yml", ".yaml"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError:
                errors.append(f"text file is not UTF-8: {name}")
                continue
            if ABSOLUTE_LOCAL_PATH.search(text):
                errors.append(f"possible machine-specific absolute path: {name}")
            if TOKEN.search(text):
                errors.append(f"possible access token: {name}")
            if path.suffix == ".json":
                try:
                    json.loads(text)
                except json.JSONDecodeError:
                    errors.append(f"invalid JSON: {name}")
    for probe in ("data/example.csv", "processed/example.npz", "artifacts/model.pt", "runs/run1/result.json", "configs/paths.local.json", ".env"):
        result = subprocess.run(command + ["check-ignore", "--quiet", "--no-index", probe])
        if result.returncode != 0:
            errors.append(f"ignore policy missing: {probe}")
    if errors:
        print("FAIL\n" + "\n".join(f"- {item}" for item in errors))
        return 1
    print(f"PASS: checked {len(names)} repository files; no real dataset or model is required")
    print("Scope: basic content policy only; human privacy/license review remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
