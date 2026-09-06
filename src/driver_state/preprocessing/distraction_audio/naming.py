"""DCPT original-clip filename parsing for the distraction audio route.

Confirmed schema
----------------
An audio census of ``First_person_view_audio.zip`` (1080/1080 files) matches::

    <task>_P<person>_<date>_<hour>_<minute>_<second>_<takeover>.<ext>

* ``task``: NDRT id ``01``-``09`` mapped onto the fixed nine-class order
  (``driver_state.constants.DCPT_CLASSES``). The task id is the label source.
* ``person``: participant ``P01``-``P40``.
* ``date``: recording date ``YYYYMMDD``.
* ``hour/minute/second``: local recording start time (the published README
  omits the seconds field; real files always contain it).
* ``takeover``: takeover time in 0.1 s units. Informational / pairing only and
  never a model feature.

The full stem (filename without the extension) is the main identifier shared by
the paired DCPT modalities (first-person audio/video, upper-body video), so it is
the ``sample_id`` for the audio metadata. Cross-modal equality for every clip is
still verified under P04 with the video owner before the identifier is frozen;
until then the schema is ``dcpt_audio_naming_v1_provisional``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from driver_state.constants import DCPT_CLASSES

NAMING_SCHEMA_VERSION = "dcpt_audio_naming_v1_provisional"

# P01 through P40 (kept in sync with the public metadata validator).
SUBJECT_PATTERN = re.compile(r"P(?:0[1-9]|[1-3][0-9]|40)")

DCPT_FILENAME_PATTERN = re.compile(
    r"^(?P<task>\d{2})_(?P<subject>P\d{2})_"
    r"(?P<date>\d{8})_(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_"
    r"(?P<takeover>\d+)\."
    r"(?P<ext>[A-Za-z0-9]+)$"
)


@dataclass(frozen=True)
class DcptClipRef:
    """Parsed identity of one DCPT original clip (audio side)."""

    task_code: int
    subject_id: str
    date: str
    start_hhmmss: str
    takeover_decisec: int
    stem: str
    extension: str

    @property
    def session_id(self) -> str:
        """Recording start instant; one takeover trial maps to one session."""
        return f"{self.date}_{self.start_hhmmss}"

    @property
    def label_id(self) -> int:
        return self.task_code - 1

    @property
    def label_class(self) -> str:
        return DCPT_CLASSES[self.label_id]


def parse_clip_filename(name: str) -> DcptClipRef | None:
    """Parse a bare filename or an archive/directory-relative path.

    Returns ``None`` when the name does not match the confirmed DCPT naming
    schema or carries an out-of-range task / subject value. Parsing never
    guesses: non-matching names are surfaced by the audit as unparsed files.
    """
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    match = DCPT_FILENAME_PATTERN.match(base)
    if match is None:
        return None
    task = int(match.group("task"))
    subject = match.group("subject")
    if not 1 <= task <= len(DCPT_CLASSES):
        return None
    if not SUBJECT_PATTERN.fullmatch(subject):
        return None
    return DcptClipRef(
        task_code=task,
        subject_id=subject,
        date=match.group("date"),
        start_hhmmss=match.group("hour") + match.group("minute") + match.group("second"),
        takeover_decisec=int(match.group("takeover")),
        stem=base[: base.rindex(".")],
        extension=match.group("ext").lower(),
    )