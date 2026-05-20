# Whisper.cpp Wrapper API

FastAPI wrapper around `whisper.cpp` for queued transcription jobs.

## Configuration

Set `WHISPERCPP_BASE_DIR` to the local `whisper.cpp` checkout:

```env
WHISPERCPP_BASE_DIR=/path/to/whisper.cpp
WHISPERCPP_CHUNK_SECONDS=0
WHISPERCPP_CHUNK_OVERLAP_SECONDS=30
WHISPERCPP_STITCH_METHOD=fuzzy
WHISPERCPP_REPETITION_GUARD=true
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
  "chunk_seconds": 1800,
  "chunk_overlap_seconds": 30,
  "stitch_method": "fuzzy",
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
- `chunk_seconds > 0`: split the audio into chunks and combine the chunk results.

When chunking is enabled, the wrapper now tries silence-aligned chunking first:

1. It runs `whisper-vad-speech-segments` once on the full input using the same VAD settings passed to the API.
2. It infers silence gaps from the detected speech segments.
3. It searches for a safe silence cut near each absolute target: `chunk_seconds`, `2 * chunk_seconds`, `3 * chunk_seconds`, and so on.
4. It selects the silence cut closest to each target.
5. It extracts non-overlapping chunks at those silence cuts and combines the resulting segments in order.

The targets are absolute. For example, with `chunk_seconds = 300`, target cuts stay near `300`, `600`, `900`, etc. If the first selected cut moves to `400`, the next target is still `600`, not `700`.

For successful silence-aligned chunking, the effective overlap is `0`, even if `chunk_overlap_seconds` was requested. The original requested overlap is still reported in result metadata as `requested_overlap_seconds`.

## Fixed Fallback

If the VAD helper is unavailable, fails, produces unparsable output, or cannot find safe silence cuts, the wrapper falls back to the previous fixed chunking behavior.

In fallback mode:

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
  "overlap_seconds": 0,
  "requested_overlap_seconds": 30,
  "strategy": "vad_silence",
  "silence_cuts": [
    {
      "target_seconds": 1800.0,
      "selected_seconds": 1798.42,
      "silence_start_seconds": 1797.91,
      "silence_end_seconds": 1798.93,
      "distance_seconds": 1.58
    }
  ],
  "chunk_count": 2,
  "stitch_method": "fuzzy",
  "stitch_methods": ["fuzzy"],
  "repetition_guard": true
}
```

Possible `strategy` values:

- `vad_silence`: silence-aligned chunks were used.
- `fixed_fallback`: silence planning failed, so fixed timestamp chunks were used.
- `fixed`: fixed chunking was used because silence planning was not needed, such as a single-chunk input.

## Local Checks

Use the project virtualenv for tests:

```bash
.venv/bin/python -m unittest discover -v
```
