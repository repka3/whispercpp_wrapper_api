import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import Settings


ProgressCallback = Callable[[int], None]


class TranscriptionError(RuntimeError):
    pass


def probe_duration_seconds(input_path: Path) -> float:
    if not shutil.which("ffprobe"):
        return 0.0

    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    try:
        return max(float(proc.stdout.strip()), 0.0)
    except ValueError:
        return 0.0


def normalize_whisper_json(
    *,
    raw_path: Path,
    job_id: str,
    settings: Settings,
    language: str,
    beam_size: int,
    best_of: int,
    vad_threshold: float,
    vad_max_speech_duration_s: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    audio_duration_seconds: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    with raw_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    source_segments = raw.get("transcription") or raw.get("segments") or []
    segments: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for item in source_segments:
        segment_text = (item.get("text") or item.get("transcript") or "").strip()
        start, end = _extract_segment_times(item)
        if segment_text:
            text_parts.append(segment_text)
        segments.append(
            {
                "start": start,
                "end": end,
                "transcript": segment_text,
                "words": [],
            }
        )

    text = (raw.get("text") or " ".join(text_parts)).strip()
    rtf = elapsed_seconds / audio_duration_seconds if audio_duration_seconds > 0 else 0.0
    speedup = audio_duration_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0

    return {
        "job_id": job_id,
        "engine": "whisper.cpp",
        "model": settings.whispercpp_model.name,
        "language": language,
        "text": text,
        "segments": segments,
        "vad": {
            "enabled": True,
            "threshold": vad_threshold,
            "max_speech_duration_seconds": vad_max_speech_duration_s,
            "min_silence_duration_ms": vad_min_silence_duration_ms,
            "speech_pad_ms": vad_speech_pad_ms,
        },
        "decode": {
            "beam_size": beam_size,
            "best_of": best_of,
        },
        "metrics": {
            "audio_duration_seconds": round(audio_duration_seconds, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "rtf": round(rtf, 4),
            "speedup": round(speedup, 4),
        },
    }


def run_transcription(
    *,
    job_id: str,
    job_dir: Path,
    input_path: Path,
    settings: Settings,
    language: str,
    beam_size: int,
    best_of: int,
    vad_threshold: float,
    vad_max_speech_duration_s: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    set_progress: ProgressCallback,
) -> dict[str, Any]:
    stdout_log = job_dir / "whisper_stdout.log"
    stderr_log = job_dir / "whisper_stderr.log"
    output_base = job_dir / "whisper_output"
    raw_json = output_base.with_suffix(".json")

    started = time.monotonic()
    audio_duration_seconds = probe_duration_seconds(input_path)
    set_progress(5)
    proc = _start_whisper(
        input_path=input_path,
        output_base=output_base,
        settings=settings,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        vad_threshold=vad_threshold,
        vad_max_speech_duration_s=vad_max_speech_duration_s,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
    )
    returncode = _capture_process(proc, stdout_log, stderr_log, set_progress)

    if returncode != 0:
        converted = _maybe_convert_to_wav(input_path, job_dir)
        if converted is not None:
            set_progress(5)
            output_base = job_dir / "whisper_output_retry"
            raw_json = output_base.with_suffix(".json")
            with stderr_log.open("ab") as handle:
                handle.write(b"\nRetrying with ffmpeg-converted 16 kHz mono WAV.\n")
            proc = _start_whisper(
                input_path=converted,
                output_base=output_base,
                settings=settings,
                language=language,
                beam_size=beam_size,
                best_of=best_of,
                vad_threshold=vad_threshold,
                vad_max_speech_duration_s=vad_max_speech_duration_s,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
            )
            returncode = _capture_process(proc, stdout_log, stderr_log, set_progress)

    if returncode != 0:
        raise TranscriptionError(_tail_text(stderr_log) or f"whisper-cli exited with code {returncode}")
    if not raw_json.exists():
        raise TranscriptionError("whisper-cli completed but did not produce JSON output")

    set_progress(95)
    elapsed_seconds = time.monotonic() - started
    result = normalize_whisper_json(
        raw_path=raw_json,
        job_id=job_id,
        settings=settings,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        vad_threshold=vad_threshold,
        vad_max_speech_duration_s=vad_max_speech_duration_s,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        audio_duration_seconds=audio_duration_seconds,
        elapsed_seconds=elapsed_seconds,
    )
    result_path = job_dir / "result.json"
    _write_json(result_path, result)
    return result


def _start_whisper(
    *,
    input_path: Path,
    output_base: Path,
    settings: Settings,
    language: str,
    beam_size: int,
    best_of: int,
    vad_threshold: float,
    vad_max_speech_duration_s: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
) -> subprocess.Popen[bytes]:
    command = [
        str(settings.whispercpp_bin),
        "--model",
        str(settings.whispercpp_model),
        "--file",
        str(input_path),
        "--language",
        language,
        "--vad",
        "--vad-model",
        str(settings.whispercpp_vad_model),
        "--vad-threshold",
        f"{vad_threshold:g}",
        "--vad-max-speech-duration-s",
        str(vad_max_speech_duration_s),
        "--vad-min-silence-duration-ms",
        str(vad_min_silence_duration_ms),
        "--vad-speech-pad-ms",
        str(vad_speech_pad_ms),
        "--beam-size",
        str(beam_size),
        "--best-of",
        str(best_of),
        "--output-json",
        "--output-json-full",
        "--print-progress",
        "--output-file",
        str(output_base),
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def _capture_process(
    proc: subprocess.Popen[bytes],
    stdout_log: Path,
    stderr_log: Path,
    set_progress: ProgressCallback,
) -> int:
    threads = [
        threading.Thread(target=_stream_pipe, args=(proc.stdout, stdout_log, set_progress), daemon=True),
        threading.Thread(target=_stream_pipe, args=(proc.stderr, stderr_log, set_progress), daemon=True),
    ]
    for thread in threads:
        thread.start()

    returncode = proc.wait()
    for thread in threads:
        thread.join(timeout=2)
    return returncode


def _stream_pipe(pipe: Any, log_path: Path, set_progress: ProgressCallback) -> None:
    if pipe is None:
        return

    progress_re = re.compile(rb"(\d{1,3})\s*%")
    tail = b""
    with log_path.open("ab") as log:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            os.fsync(log.fileno())
            scan = tail + chunk
            for match in progress_re.finditer(scan):
                progress = max(5, min(94, int(match.group(1))))
                set_progress(progress)
            tail = scan[-32:]


def _maybe_convert_to_wav(input_path: Path, job_dir: Path) -> Path | None:
    if not shutil.which("ffmpeg"):
        return None

    converted = job_dir / "converted_16khz_mono.wav"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(converted),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode == 0 and converted.exists():
        return converted
    return None


def _extract_segment_times(item: dict[str, Any]) -> tuple[float, float]:
    timestamps = item.get("timestamps")
    if isinstance(timestamps, dict):
        return _as_seconds(timestamps.get("from")), _as_seconds(timestamps.get("to"))

    offsets = item.get("offsets")
    if isinstance(offsets, dict):
        return _as_milliseconds(offsets.get("from")), _as_milliseconds(offsets.get("to"))

    return _as_seconds(item.get("start") or item.get("t0")), _as_seconds(item.get("end") or item.get("t1"))


def _as_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    if isinstance(value, str):
        parts = value.replace(",", ".").split(":")
        try:
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 3)
            return round(float(value), 3)
        except ValueError:
            return 0.0
    return 0.0


def _as_milliseconds(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return round(float(value) / 1000.0, 3)
    except (TypeError, ValueError):
        return 0.0


def _tail_text(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()[-limit:]
    return data.decode("utf-8", errors="replace").strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)
