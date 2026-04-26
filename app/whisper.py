import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Settings


ProgressCallback = Callable[[int], None]


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkWindow:
    index: int
    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class RepetitionReport:
    detected: bool
    first_bad_index: int | None = None
    phrase: str = ""
    count: int = 0


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
    chunking: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    with raw_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    segments = _extract_segments(raw)
    text_parts = [item["transcript"] for item in segments if item["transcript"]]

    text = (raw.get("text") or " ".join(text_parts)).strip()
    rtf = elapsed_seconds / audio_duration_seconds if audio_duration_seconds > 0 else 0.0
    speedup = audio_duration_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
    decode = {
        "beam_size": beam_size,
        "best_of": best_of,
    }
    if chunking is not None:
        decode["chunking"] = chunking

    result = {
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
        "decode": decode,
        "metrics": {
            "audio_duration_seconds": round(audio_duration_seconds, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "rtf": round(rtf, 4),
            "speedup": round(speedup, 4),
        },
    }
    if warnings:
        result["warnings"] = warnings
    return result


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
    chunking_mode: str,
    chunk_seconds: int,
    chunk_overlap_seconds: int,
    repetition_guard: bool,
    set_progress: ProgressCallback,
) -> dict[str, Any]:
    audio_duration_seconds = probe_duration_seconds(input_path)
    chunking_mode = chunking_mode if chunking_mode in {"off", "auto", "always"} else settings.chunking_mode
    chunk_seconds = max(int(chunk_seconds), 1)
    chunk_overlap_seconds = max(int(chunk_overlap_seconds), 0)
    if chunk_overlap_seconds >= chunk_seconds:
        chunk_overlap_seconds = max(chunk_seconds - 1, 0)

    if _should_chunk(
        mode=chunking_mode,
        audio_duration_seconds=audio_duration_seconds,
        threshold_seconds=settings.chunk_threshold_seconds,
    ):
        return _run_chunked_transcription(
            job_id=job_id,
            job_dir=job_dir,
            input_path=input_path,
            settings=settings,
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            vad_threshold=vad_threshold,
            vad_max_speech_duration_s=vad_max_speech_duration_s,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            chunking_mode=chunking_mode,
            chunk_seconds=chunk_seconds,
            chunk_overlap_seconds=chunk_overlap_seconds,
            repetition_guard=repetition_guard,
            audio_duration_seconds=audio_duration_seconds,
            set_progress=set_progress,
        )

    return _run_single_transcription(
        job_id=job_id,
        job_dir=job_dir,
        input_path=input_path,
        settings=settings,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        vad_threshold=vad_threshold,
        vad_max_speech_duration_s=vad_max_speech_duration_s,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        audio_duration_seconds=audio_duration_seconds,
        set_progress=set_progress,
    )


def _run_single_transcription(
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
    audio_duration_seconds: float,
    set_progress: ProgressCallback,
) -> dict[str, Any]:
    stdout_log = job_dir / "whisper_stdout.log"
    stderr_log = job_dir / "whisper_stderr.log"
    output_base = job_dir / "whisper_output"
    raw_json = output_base.with_suffix(".json")

    started = time.monotonic()
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


def _run_chunked_transcription(
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
    chunking_mode: str,
    chunk_seconds: int,
    chunk_overlap_seconds: int,
    repetition_guard: bool,
    audio_duration_seconds: float,
    set_progress: ProgressCallback,
) -> dict[str, Any]:
    started = time.monotonic()
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = job_dir / "whisper_stdout.log"
    stderr_log = job_dir / "whisper_stderr.log"
    stitch_debug_path = job_dir / "stitch_debug.md"
    stitch_debug_json_path = job_dir / "stitch_debug.json"
    windows = build_chunk_windows(
        audio_duration_seconds=audio_duration_seconds,
        chunk_seconds=chunk_seconds,
        overlap_seconds=chunk_overlap_seconds,
    )
    if not windows:
        raise TranscriptionError("Unable to build transcription chunks")

    all_segments: list[dict[str, Any]] = []
    stitch_audits: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    retried_chunks: list[int] = []
    set_progress(5)
    _append_text(stdout_log, f"Chunked transcription enabled: {len(windows)} chunks.\n")
    _append_text(stderr_log, "Chunked mode writes detailed whisper logs under chunks/<index>/.\n")
    _write_stitch_debug_header(stitch_debug_path, job_id=job_id)

    for window in windows:
        chunk_dir = chunks_dir / f"{window.index:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_input = chunk_dir / "input_16khz_mono.wav"
        _append_text(
            stdout_log,
            (
                f"Chunk {window.index:04d}: start={window.start_seconds:.3f}s "
                f"duration={window.duration_seconds:.3f}s dir={chunk_dir}\n"
            ),
        )
        _extract_audio_chunk(
            input_path=input_path,
            output_path=chunk_input,
            start_seconds=window.start_seconds,
            duration_seconds=window.duration_seconds,
        )

        raw_path, retried, warning = _run_chunk_window(
            chunk_dir=chunk_dir,
            input_path=chunk_input,
            settings=settings,
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            vad_threshold=vad_threshold,
            vad_max_speech_duration_s=vad_max_speech_duration_s,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            repetition_guard=repetition_guard,
            set_progress=lambda progress, current=window.index, total=len(windows): set_progress(
                _chunk_progress(current, total, progress)
            ),
        )
        if retried:
            retried_chunks.append(window.index)
        if warning is not None:
            warnings.append(warning)

        with raw_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        chunk_segments = _extract_segments(
            raw,
            offset_seconds=window.start_seconds,
            clamp_start=window.start_seconds,
            clamp_end=window.start_seconds + window.duration_seconds,
        )
        if warning and warning.get("first_bad_index") is not None:
            chunk_segments = chunk_segments[: int(warning["first_bad_index"])]

        all_segments, audit = merge_chunk_segments_with_audit(
            all_segments,
            chunk_segments,
            previous_chunk_index=window.index - 1,
            next_chunk_index=window.index,
            overlap_start_seconds=window.start_seconds,
            overlap_end_seconds=window.start_seconds + chunk_overlap_seconds if window.index > 0 else None,
            incoming_warning=warning,
        )
        if audit is not None:
            stitch_audits.append(audit)
            _append_text(stitch_debug_path, render_stitch_audit_markdown(audit))

    set_progress(95)
    all_segments.sort(key=lambda item: (item["start"], item["end"]))
    elapsed_seconds = time.monotonic() - started
    text = " ".join(item["transcript"] for item in all_segments if item["transcript"]).strip()
    rtf = elapsed_seconds / audio_duration_seconds if audio_duration_seconds > 0 else 0.0
    speedup = audio_duration_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
    result = {
        "job_id": job_id,
        "engine": "whisper.cpp",
        "model": settings.whispercpp_model.name,
        "language": language,
        "text": text,
        "segments": all_segments,
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
            "chunking": {
                "enabled": True,
                "mode": chunking_mode,
                "chunk_seconds": chunk_seconds,
                "overlap_seconds": chunk_overlap_seconds,
                "chunk_count": len(windows),
                "threshold_seconds": settings.chunk_threshold_seconds,
                "repetition_guard": repetition_guard,
                "retried_chunks": retried_chunks,
                "warning_count": len(warnings),
                "stitch_debug_path": str(stitch_debug_path),
                "stitch_debug_json_path": str(stitch_debug_json_path),
            },
        },
        "metrics": {
            "audio_duration_seconds": round(audio_duration_seconds, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "rtf": round(rtf, 4),
            "speedup": round(speedup, 4),
        },
    }
    if warnings:
        result["warnings"] = warnings
    _write_json(
        stitch_debug_json_path,
        {
            "job_id": job_id,
            "chunk_count": len(windows),
            "overlap_seconds": chunk_overlap_seconds,
            "boundaries": stitch_audits,
        },
    )
    result_path = job_dir / "result.json"
    _write_json(result_path, result)
    return result


def _run_chunk_window(
    *,
    chunk_dir: Path,
    input_path: Path,
    settings: Settings,
    language: str,
    beam_size: int,
    best_of: int,
    vad_threshold: float,
    vad_max_speech_duration_s: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    repetition_guard: bool,
    set_progress: ProgressCallback,
) -> tuple[Path, bool, dict[str, Any] | None]:
    output_base = chunk_dir / "whisper_output"
    raw_json = output_base.with_suffix(".json")
    stdout_log = chunk_dir / "whisper_stdout.log"
    stderr_log = chunk_dir / "whisper_stderr.log"
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
        raise TranscriptionError(_tail_text(stderr_log) or f"chunk {chunk_dir.name} exited with code {returncode}")
    if not raw_json.exists():
        raise TranscriptionError(f"chunk {chunk_dir.name} completed but did not produce JSON output")

    report = _detect_repetition_in_json(raw_json) if repetition_guard else RepetitionReport(False)
    if not report.detected:
        return raw_json, False, None

    retry_base = chunk_dir / "whisper_output_retry"
    retry_json = retry_base.with_suffix(".json")
    retry_stdout = chunk_dir / "whisper_stdout_retry.log"
    retry_stderr = chunk_dir / "whisper_stderr_retry.log"
    proc = _start_whisper(
        input_path=input_path,
        output_base=retry_base,
        settings=settings,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        vad_threshold=vad_threshold,
        vad_max_speech_duration_s=vad_max_speech_duration_s,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        safe_decode=True,
    )
    retry_code = _capture_process(proc, retry_stdout, retry_stderr, set_progress)
    if retry_code == 0 and retry_json.exists():
        retry_report = _detect_repetition_in_json(retry_json)
        if not retry_report.detected:
            return retry_json, True, {
                "chunk": int(chunk_dir.name),
                "type": "repetition_retry_succeeded",
                "phrase": report.phrase,
                "count": report.count,
            }
        report = retry_report
        raw_json = retry_json

    return raw_json, True, {
        "chunk": int(chunk_dir.name),
        "type": "repetition_kept_prefix",
        "phrase": report.phrase,
        "count": report.count,
        "first_bad_index": report.first_bad_index,
    }


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
    safe_decode: bool = False,
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
    if safe_decode:
        command.extend(["--max-context", "0", "--no-fallback"])
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


def build_chunk_windows(
    *,
    audio_duration_seconds: float,
    chunk_seconds: int,
    overlap_seconds: int,
) -> list[ChunkWindow]:
    if audio_duration_seconds <= 0:
        return []
    chunk_seconds = max(int(chunk_seconds), 1)
    overlap_seconds = max(int(overlap_seconds), 0)
    if overlap_seconds >= chunk_seconds:
        overlap_seconds = max(chunk_seconds - 1, 0)
    step = chunk_seconds - overlap_seconds
    windows: list[ChunkWindow] = []
    start = 0.0
    index = 0
    while start < audio_duration_seconds:
        duration = min(float(chunk_seconds), audio_duration_seconds - start)
        if duration <= 0:
            break
        windows.append(ChunkWindow(index=index, start_seconds=round(start, 3), duration_seconds=round(duration, 3)))
        if start + duration >= audio_duration_seconds:
            break
        start += step
        index += 1
    return windows


def merge_chunk_segments(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    overlap_start_seconds: float,
    overlap_end_seconds: float | None,
) -> list[dict[str, Any]]:
    merged, _audit = merge_chunk_segments_with_audit(
        existing,
        incoming,
        previous_chunk_index=None,
        next_chunk_index=None,
        overlap_start_seconds=overlap_start_seconds,
        overlap_end_seconds=overlap_end_seconds,
        incoming_warning=None,
    )
    return merged


def merge_chunk_segments_with_audit(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    previous_chunk_index: int | None,
    next_chunk_index: int | None,
    overlap_start_seconds: float,
    overlap_end_seconds: float | None,
    incoming_warning: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not existing or overlap_end_seconds is None:
        return existing + incoming, None

    recent_texts = {
        _normalize_for_compare(item["transcript"])
        for item in existing[-20:]
        if _normalize_for_compare(item["transcript"])
    }
    merged = list(existing)
    incoming_decisions: list[dict[str, Any]] = []
    dropped_segments: list[dict[str, Any]] = []
    kept_segments: list[dict[str, Any]] = []
    kept_overlap_segments: list[dict[str, Any]] = []
    kept_outside_overlap_segments: list[dict[str, Any]] = []
    for segment in incoming:
        segment_text = _normalize_for_compare(segment["transcript"])
        inside_overlap = segment["end"] <= overlap_end_seconds and segment["start"] >= overlap_start_seconds
        if inside_overlap and segment_text in recent_texts:
            dropped_segments.append(_audit_segment(segment))
            incoming_decisions.append(
                {
                    "decision": "dropped_duplicate",
                    "segment": _audit_segment(segment),
                }
            )
            continue
        decision = "kept" if inside_overlap else "kept_outside_overlap"
        audited = _audit_segment(segment)
        kept_segments.append(audited)
        if inside_overlap:
            kept_overlap_segments.append(audited)
        else:
            kept_outside_overlap_segments.append(audited)
        incoming_decisions.append(
            {
                "decision": decision,
                "segment": audited,
            }
        )
        merged.append(segment)
        if segment_text:
            recent_texts.add(segment_text)

    audit = {
        "previous_chunk": previous_chunk_index,
        "next_chunk": next_chunk_index,
        "overlap_start": round(overlap_start_seconds, 3),
        "overlap_end": round(overlap_end_seconds, 3),
        "overlap_start_label": format_timestamp(overlap_start_seconds),
        "overlap_end_label": format_timestamp(overlap_end_seconds),
        "previous_tail": [_audit_segment(item) for item in existing[-8:]],
        "incoming_head": incoming_decisions[:8],
        "dropped_duplicates": dropped_segments,
        "kept_overlap": kept_overlap_segments,
        "kept_outside_overlap_head": kept_outside_overlap_segments[:8],
        "incoming_warning": incoming_warning,
        "counts": {
            "incoming": len(incoming),
            "dropped_duplicates": len(dropped_segments),
            "kept": len(kept_segments),
            "kept_overlap": len(kept_overlap_segments),
            "kept_outside_overlap": len(kept_outside_overlap_segments),
        },
    }
    return merged, audit


def render_stitch_audit_markdown(audit: dict[str, Any]) -> str:
    previous_chunk = _chunk_label(audit.get("previous_chunk"))
    next_chunk = _chunk_label(audit.get("next_chunk"))
    counts = audit.get("counts", {})
    lines = [
        "",
        f"## Chunk {previous_chunk} -> {next_chunk}",
        "",
        f"Overlap: {audit['overlap_start_label']} -> {audit['overlap_end_label']}",
        "",
        "Previous tail:",
    ]
    lines.extend(_render_segment_lines(audit.get("previous_tail") or []))
    lines.extend(["", "Incoming head:"])
    for item in audit.get("incoming_head") or []:
        segment = item["segment"]
        lines.append(
            f"- {item['decision']} [{segment['start_label']} -> {segment['end_label']}] {segment['transcript']}"
        )
    if not audit.get("incoming_head"):
        lines.append("- none")

    warning = audit.get("incoming_warning")
    if warning:
        lines.extend(
            [
                "",
                "Incoming warning:",
                f"- {warning.get('type', 'warning')} chunk={warning.get('chunk')} phrase={warning.get('phrase', '')}",
            ]
        )

    lines.extend(
        [
            "",
            "Decision:",
            (
                f"Dropped {counts.get('dropped_duplicates', 0)} duplicate segment(s) from overlap, "
                f"kept {counts.get('kept', 0)} incoming segment(s) "
                f"({counts.get('kept_overlap', 0)} inside overlap, "
                f"{counts.get('kept_outside_overlap', 0)} outside overlap)."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _render_segment_lines(segments: list[dict[str, Any]]) -> list[str]:
    if not segments:
        return ["- none"]
    return [
        f"- [{segment['start_label']} -> {segment['end_label']}] {segment['transcript']}"
        for segment in segments
    ]


def _audit_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": segment["start"],
        "end": segment["end"],
        "start_label": format_timestamp(segment["start"]),
        "end_label": format_timestamp(segment["end"]),
        "transcript": segment["transcript"],
    }


def _chunk_label(value: Any) -> str:
    if isinstance(value, int) and value >= 0:
        return f"{value:04d}"
    return "n/a"


def format_timestamp(seconds: float) -> str:
    millis_total = max(int(round(seconds * 1000)), 0)
    millis = millis_total % 1000
    total_seconds = millis_total // 1000
    secs = total_seconds % 60
    minutes_total = total_seconds // 60
    mins = minutes_total % 60
    hours = minutes_total // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def _should_chunk(*, mode: str, audio_duration_seconds: float, threshold_seconds: int) -> bool:
    if mode == "off":
        return False
    if mode == "always":
        return audio_duration_seconds > 0
    return audio_duration_seconds >= threshold_seconds


def _chunk_progress(chunk_index: int, chunk_count: int, chunk_progress: int) -> int:
    chunk_count = max(chunk_count, 1)
    chunk_progress = max(0, min(chunk_progress, 100))
    return max(5, min(94, int(5 + ((chunk_index + chunk_progress / 100.0) / chunk_count) * 90)))


def _extract_audio_chunk(
    *,
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    if not shutil.which("ffmpeg"):
        raise TranscriptionError("ffmpeg is required for chunked transcription")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0 or not output_path.exists():
        message = proc.stderr.decode("utf-8", errors="replace")[-4000:].strip()
        raise TranscriptionError(message or f"ffmpeg failed while creating chunk at {start_seconds:.3f}s")


def _extract_segments(
    raw: dict[str, Any],
    *,
    offset_seconds: float = 0.0,
    clamp_start: float | None = None,
    clamp_end: float | None = None,
) -> list[dict[str, Any]]:
    source_segments = raw.get("transcription") or raw.get("segments") or []
    segments: list[dict[str, Any]] = []
    for item in source_segments:
        segment_text = (item.get("text") or item.get("transcript") or "").strip()
        start, end = _extract_segment_times(item)
        start = start + offset_seconds
        end = end + offset_seconds
        if clamp_start is not None:
            start = max(start, clamp_start)
        if clamp_end is not None:
            end = min(end, clamp_end)
        if end < start:
            end = start
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "transcript": segment_text,
                "words": [],
            }
        )
    return segments


def _detect_repetition_in_json(raw_path: Path) -> RepetitionReport:
    with raw_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return detect_repetition(_extract_segments(raw))


def detect_repetition(segments: list[dict[str, Any]]) -> RepetitionReport:
    previous = ""
    count = 0
    first_bad_index: int | None = None
    for index, segment in enumerate(segments):
        phrase = _normalize_for_compare(segment.get("transcript", ""))
        if not _is_repetition_candidate(phrase):
            previous = ""
            count = 0
            first_bad_index = None
            continue
        if phrase == previous:
            count += 1
            if count == 2:
                first_bad_index = index - 1
            if count >= 4:
                return RepetitionReport(
                    detected=True,
                    first_bad_index=first_bad_index,
                    phrase=phrase,
                    count=count,
                )
        else:
            previous = phrase
            count = 1
            first_bad_index = None
    return RepetitionReport(False)


def _is_repetition_candidate(phrase: str) -> bool:
    if len(phrase) < 10:
        return False
    return len(phrase.split()) >= 3


def _normalize_for_compare(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


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


def _append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _write_stitch_debug_header(path: Path, *, job_id: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Stitch Debug\n\n")
        handle.write(f"Job: `{job_id}`\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)
