# Whisper.cpp Wrapper API

FastAPI wrapper around `whisper.cpp` for queued transcription jobs.

## Configuration

Set `WHISPERCPP_BASE_DIR` to the local `whisper.cpp` checkout:

```env
WHISPERCPP_BASE_DIR=/path/to/whisper.cpp
WHISPERCPP_CHUNK_SECONDS=1800
WHISPERCPP_CHUNK_OVERLAP_SECONDS=30
WHISPERCPP_STITCH_METHOD=center_align
WHISPERCPP_REPETITION_GUARD=true
WHISPERCPP_VAD_CUT_THRESHOLD=0.5
```

The wrapper expects:

- `build/bin/whisper-cli`
- `build/bin/whisper-vad-speech-segments`
- transcription models in `models/`
- a Silero VAD model such as `models/ggml-silero-v6.2.0.bin`

## Transcription API

Create a job from an existing local path:

```http
POST /jobs/transcribe/path
Content-Type: application/json
```

```json
{
  "path": "/absolute/path/audio.mp3",
  "model": "ggml-large-v3-turbo.bin",
  "language": "it",
  "vad_threshold": 0.01,
  "vad_cut_threshold": 0.5,
  "chunk_seconds": 1800,
  "chunk_overlap_seconds": 30,
  "stitch_method": "center_align",
  "repetition_guard": true
}
```

Create a job from an upload:

```http
POST /jobs/transcribe/upload
Content-Type: multipart/form-data
```

Use the same fields as the path API, plus the uploaded `file`.

Poll a job:

```http
GET /jobs/{job_id}
```

List completed or failed jobs:

```http
GET /jobs
```

## Chunking Behavior

`chunk_seconds` controls whether chunking is enabled.

- `chunk_seconds = 0`: run one full-file transcription pass.
- `chunk_seconds > 0`: split the audio into chunks and combine the chunk results only when the audio is longer than `chunk_seconds`.

The default is `1800` seconds. Files up to and including that duration still use the same single-pass path as `chunk_seconds = 0`: no chunk-planning VAD pre-pass, no temporary chunk extraction, and no stitching.

`vad_threshold` is used by `whisper-cli` while transcribing each full file or chunk.
`vad_cut_threshold` is used only for the pre-cut planning pass. This lets transcription stay sensitive, for example `0.01`, while cut planning uses a stricter threshold, for example `0.5`, so silence gaps are easier to find.

When chunking is enabled, the wrapper now tries silence-aligned cuts at each target boundary:

1. It runs `whisper-vad-speech-segments` once on the full input using `vad_cut_threshold`.
2. It infers silence gaps from the detected speech segments.
3. It searches for a safe silence cut near each absolute target: `chunk_seconds`, `2 * chunk_seconds`, `3 * chunk_seconds`, and so on.
4. For each target, it selects the silence cut closest to that target when one is usable.
5. If one target has no usable silence, only that boundary becomes a fixed hard fallback cut.
6. It extracts chunks and combines the resulting segments in order.

The targets are absolute. For example, with `chunk_seconds = 300`, target cuts stay near `300`, `600`, `900`, etc. If the first selected cut moves to `400`, the next target is still `600`, not `700`.

VAD-silence boundaries and hard fallback boundaries both use `chunk_overlap_seconds` as leading context for the next chunk. This gives the stitcher real duplicated audio to align while still selecting the logical cut from VAD silence when possible. The original requested overlap is reported in result metadata as `requested_overlap_seconds`.

Available stitch methods:

- `fuzzy`: drops or trims duplicate incoming segments by normalized token overlap.
- `safe_zone`: keeps only the middle of each overlapped chunk window by segment midpoint.
- `word_align`: aligns word timestamps across the boundary and trims the duplicate incoming word prefix, falling back to `fuzzy` if word timings are unavailable.
- `center_align`: finds a shared word sequence near the overlap center, trims the previous chunk after that seam and the next chunk before it, then runs fuzzy dedupe on the remaining overlap. This is the default because it performed best overall in the golden WER runs and in the synthetic long-meeting run.

## Fixed Fallback

If the VAD helper is unavailable, fails, or produces unparsable output, the wrapper falls back to the previous fixed chunking behavior for the whole file.

In full fallback mode:

- chunks are cut at fixed timestamps;
- `chunk_overlap_seconds` is honored;
- existing stitching behavior is used;
- the result includes a warning with type `chunking_fixed_fallback`;
- logs include a chunk planning warning.

## Result Metadata

Chunked results include `decode.chunking` metadata:

```json
{
  "enabled": true,
  "chunk_seconds": 1800,
  "overlap_seconds": 30,
  "requested_overlap_seconds": 30,
  "strategy": "vad_silence",
  "silence_cuts": [
    {
      "target_seconds": 1800.0,
      "selected_seconds": 1798.42,
      "silence_start_seconds": 1797.91,
      "silence_end_seconds": 1798.93,
      "distance_seconds": 1.58,
      "cut_type": "vad_silence"
    }
  ],
  "chunk_count": 2,
  "stitch_method": "center_align",
  "stitch_methods": ["center_align"],
  "repetition_guard": true
}
```

Possible `strategy` values:

- `vad_silence`: silence-aligned chunks were used.
- `mixed`: at least one boundary used a VAD silence cut and at least one boundary used a hard fallback cut.
- `fixed_fallback`: silence planning failed, so fixed timestamp chunks were used.
- `fixed`: fixed chunking was used because silence planning was not needed, such as a single-chunk input.

Each item in `silence_cuts` includes `cut_type`:

- `vad_silence`: the boundary was moved to a detected silence gap.
- `hard_fallback`: no usable silence was found for that target, so that boundary used the target timestamp and requested overlap.

## Local Checks

Use the project virtualenv for tests:

```bash
.venv/bin/python -m unittest discover -v
```

## Golden Cut Debugging

Use the hand-checked `golden/*.mp3` + `golden/*.txt` pairs to debug chunk planning and WER regressions locally.

Fast VAD/cut planning only:

```bash
.venv/bin/python scripts/golden_cut_debug.py --planner-only
```

Full transcription comparison:

```bash
.venv/bin/python scripts/golden_cut_debug.py
```

The runner writes `summary.csv`, `summary.md`, and per-case diagnostics under `golden_runs/<timestamp>/`.
It defaults to `vad_threshold=0.01` for transcription, `vad_cut_threshold=0.5` for cut planning,
simulated chunk sizes `120 300`, and stitch variants `fuzzy safe_zone word_align center_align`.
