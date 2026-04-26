# Whisper.cpp Native Wrapper API

## Summary

Build this directory as a native Python FastAPI app in a simple venv, no Docker. The app should match the current Luna API style: queued transcription jobs, upload/path inputs, health checks, result retrieval, and a small HTML upload page.

Must be limit 1 job per time. This is an api for a single machine with a single user.

Transcription must run through `whisper-cli` as a subprocess for each job. Do not keep `whisper-server` running permanently, because it keeps the model resident in RAM. A per-job CLI process lets the OS reclaim model memory after every transcription.

## Runtime Defaults

Use this tested Jetson command shape as the default:

```bash
/home/transcribe/whisper.cpp/build/bin/whisper-cli \
  --model /home/transcribe/whisper.cpp/models/ggml-large-v3-q5_0.bin \
  --file INPUT_FILE \
  --language it \
  --vad \
  --vad-model /home/transcribe/whisper.cpp/models/ggml-silero-v6.2.0.bin \
  --vad-threshold 0.5 \
  --vad-max-speech-duration-s 30 \
  --vad-min-silence-duration-ms 2000 \
  --vad-speech-pad-ms 400 \
  --beam-size 3 \
  --best-of 3 \
  --output-json \
  --output-json-full \
  --print-progress \
  --output-file JOB_OUTPUT_BASE
```

Make these configurable through environment variables:

- `WHISPERCPP_BIN`, default `/home/transcribe/whisper.cpp/build/bin/whisper-cli`
- `WHISPERCPP_MODEL`, default `/home/transcribe/whisper.cpp/models/ggml-large-v3-q5_0.bin`
- `WHISPERCPP_VAD_MODEL`, default `/home/transcribe/whisper.cpp/models/ggml-silero-v6.2.0.bin`
- `WHISPERCPP_TEMP_DIR`, default this project directory, so jobs are stored under `./jobs`
- `WHISPERCPP_DEFAULT_LANGUAGE`, default `it`
- `WHISPERCPP_BEAM_SIZE`, default `3`
- `WHISPERCPP_BEST_OF`, default `3`

## API And Behavior

Expose the same style of API as the current Luna app:

- `GET /`: serve the HTML test/upload page.
- `GET /health`: verify `whisper-cli`, model file, VAD model file, temp directory, `ffmpeg`, and `ffprobe`.
- `POST /jobs/transcribe/upload`: accept an audio/video upload and queue a job.
- `POST /jobs/transcribe/path`: accept a local filesystem path and queue a job.
- `GET /jobs/{job_id}`: return status, progress, input metadata, and error.
- `GET /jobs/{job_id}/result`: return the final result JSON once the job is complete.

Unlike the Docker app, path mode does not need a shared-root restriction. It only needs to verify that the requested path exists and is a readable file on the device.

Use file-backed job metadata with these statuses:

- `queued`
- `running`
- `succeeded`
- `failed`

Run a single background worker thread and process one job at a time. This is intentional for Jetson memory safety.

## Implementation Details

Store job data under:

```text
WHISPERCPP_TEMP_DIR/jobs/{job_id}/
```

With the default config, this resolves to:

```text
./jobs/{job_id}/
```

Each job directory should contain:

- `metadata.json`
- uploaded source file, for upload jobs
- `whisper_stdout.log`
- `whisper_stderr.log`
- raw whisper output JSON
- normalized `result.json`

Run `whisper-cli` with `subprocess.Popen`, capture stdout/stderr continuously, and update job progress when progress output can be parsed. If progress parsing is unreliable, keep progress coarse:

- `0`: queued
- `1`: claimed/running
- `5`: subprocess started
- `95`: subprocess exited and parsing output
- `100`: succeeded

Do not pre-convert every input. `whisper-cli` supports `flac`, `mp3`, `ogg`, and `wav`. If decoding fails, optionally retry once by converting to mono 16 kHz PCM WAV with `ffmpeg`.

If `whisper-cli` exits nonzero, mark the job failed and expose the captured stderr in the job error or result payload.

## Result Shape

Normalize whisper.cpp JSON into this Luna-like response:

```json
{
  "job_id": "...",
  "engine": "whisper.cpp",
  "model": "ggml-large-v3-q5_0.bin",
  "language": "it",
  "text": "...",
  "segments": [
    {
      "start": 0.0,
      "end": 12.34,
      "transcript": "...",
      "words": []
    }
  ],
  "vad": {
    "enabled": true,
    "threshold": 0.5,
    "max_speech_duration_seconds": 30,
    "min_silence_duration_ms": 2000,
    "speech_pad_ms": 400
  },
  "decode": {
    "beam_size": 3,
    "best_of": 3
  },
  "metrics": {
    "audio_duration_seconds": 0.0,
    "elapsed_seconds": 0.0,
    "rtf": 0.0,
    "speedup": 0.0
  }
}
```

For v1, leave `words` as an empty list. Segment timestamps from whisper.cpp are enough for the first production wrapper.

## HTML Page

Create a simple test page similar to the existing Luna upload page, renamed for Whisper.cpp.

The page should include:

- audio/video file upload
- local path input
- language input, default `it`
- beam size input, default `3`
- best-of input, default `3`
- VAD threshold, default `0.5`
- VAD max speech seconds, default `30`
- VAD min silence ms, default `2000`
- VAD speech pad ms, default `400`
- health display
- queued/running status display
- final normalized JSON display
- failed-job error display

The page should submit upload jobs via `POST /jobs/transcribe/upload`, path jobs via `POST /jobs/transcribe/path`, poll `GET /jobs/{job_id}`, and fetch `GET /jobs/{job_id}/result` after success.

## Native Setup

Initial manual setup:

```bash
cd /home/transcribe/whispercpp_wrapper_api
python -m venv .venv
. .venv/bin/activate
pip install fastapi "uvicorn[standard]" python-multipart
uvicorn app.main:app --host 0.0.0.0 --port 7600
```

The app should not require PyTorch, Transformers, Docker, or CUDA Python packages.

## Test Plan

Test in this order:

1. `GET /health` reports all required binaries and model files present.
2. Upload the 10-minute sample and confirm result JSON is produced.
3. Submit a local MP3 path and confirm path mode works without any shared-directory rule.
4. Submit a 2-hour meeting while watching `tegrastats`.
5. Confirm `whisper-cli` exits after each job and RAM drops afterward.
6. Confirm the FastAPI wrapper remains alive after a failed transcription.
7. Confirm interrupted running jobs are marked failed on API restart.

## Assumptions

- Default production model is `ggml-large-v3-q5_0.bin`.
- Default VAD model is `ggml-silero-v6.2.0.bin`.
- Default language is Italian, `it`.
- One job at a time is the correct behavior for Jetson stability.
- This project is native-only and does not use containers.
