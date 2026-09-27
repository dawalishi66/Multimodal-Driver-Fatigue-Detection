"""DCPT original-clip filename parsing for the distraction-video route."""

from __future__ import annotations

import re
from dataclasses import dataclass

from driver_state.preprocessing.distraction_video.fusion_labels import (
    is_six_class_task,
    six_class_label,
)

VIDEO_NAMING_SCHEMA_VERSION = "dcpt_video_naming_v1_provisional"

SUBJECT_PATTERN = re.compile(r"P(?:0[1-9]|[1-3][0-9]|40)")
DCPT_VIDEO_PATTERN = re.compile(
    r"^(?P<task>\d{2})_(?P<subject>P\d{2})_"
    r"(?P<date>\d{8})_(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_"
    r"(?P<takeover>\d+)\.(?P<ext>mp4|avi|mov)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VideoClipRef:
    """Parsed identity of one DCPT original video clip."""

    task_code: int
    subject_id: str
    date: str
    start_hhmmss: str
    takeover_decisec: int
    stem: str
    extension: str

    @property
    def session_id(self) -> str:
        """Return the same session format used by the audio module."""
        return (
            f"{self.subject_id}_{self.date}_"
            f"{self.start_hhmmss[:4]}_{self.start_hhmmss[4:]}"
        )

    @property
    def label_id(self) -> int:
        return six_class_label(self.task_code)[0]

    @property
    def label_class(self) -> str:
        return six_class_label(self.task_code)[1]


def parse_clip_filename(name: str) -> VideoClipRef | None:
    """Parse a bare filename or relative path; return None without guessing."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    match = DCPT_VIDEO_PATTERN.fullmatch(base)
    if match is None:
        return None
    task = int(match.group("task"))
    subject = match.group("subject")
    if not is_six_class_task(task) or SUBJECT_PATTERN.fullmatch(subject) is None:
        return None
    return VideoClipRef(
        task_code=task,
        subject_id=subject,
        date=match.group("date"),
        start_hhmmss=match.group("hour") + match.group("minute") + match.group("second"),
        takeover_decisec=int(match.group("takeover")),
        stem=base[: base.rindex(".")],
        extension=match.group("ext").lower(),
    )
