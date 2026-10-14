# P04 DCPT Audio/Video Alignment Audit v1

Status: `pending_manual_signoff`

## Scope

714 video clips and 714 audio clips with matching `sample_id`.

## Automated Evidence

- Common sample IDs: 714
- Video-only IDs: 0
- Audio-only IDs: 0
- Label mismatches: 0
- Session mismatch count: 0
- Split mismatch count: 0
- Video feature SHA-256 mismatches: 0
- Audio feature SHA-256 mismatches: 0

The deterministic review queue contains three different tasks for each
subject and is recorded in
`manifests/dcpt_video_6c_v1/p04_manual_review_queue_v1.jsonl`.

## Required Manual Review

For each queued clip, reviewers must jointly inspect the original upper-body
video and first-person audio, record start/end observations, mark exceptions,
and sign both reviewer fields. Matching file stems alone are not sufficient.

## Evidence Limits

- Raw upper-body videos are unavailable in the current environment, so no new
  decoder-level playback or timestamp replay was performed here.
- No test prediction was used to choose the review queue.
- The queue is pending human signatures; this report is not a P04 PASS.

