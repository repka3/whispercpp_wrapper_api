import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from . import stitch_utils


ProgressCallback = Callable[[int], None]
StitchLogCallback = Callable[[dict[str, Any]], None]


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkWindow:
    index: int
    start_seconds: float
    duration_seconds: float
    previous_overlap_seconds: int = 0
    next_overlap_seconds: int = 0


@dataclass(frozen=True)
class VadSpeechSegment:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class SilenceCutDecision:
    target_seconds: float
    selected_seconds: float
    silence_start_seconds: float
    silence_end_seconds: float
    cut_type: str = "vad_silence"
    reason: str = ""

    @property
    def distance_seconds(self) -> float:
        return abs(self.selected_seconds - self.target_seconds)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "target_seconds": round(self.target_seconds, 3),
            "selected_seconds": round(self.selected_seconds, 3),
            "silence_start_seconds": round(self.silence_start_seconds, 3),
            "silence_end_seconds": round(self.silence_end_seconds, 3),
            "distance_seconds": round(self.distance_seconds, 3),
            "cut_type": self.cut_type,
        }
        if self.reason:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True)
class ChunkWindowPlan:
    strategy: str
    windows: list[ChunkWindow]
    overlap_seconds: int
    warnings: list[dict[str, Any]]
    cut_decisions: list[SilenceCutDecision]


@dataclass(frozen=True)
class RepetitionReport:
    detected: bool
    first_bad_index: int | None = None
    phrase: str = ""
    count: int = 0


@dataclass(frozen=True)
class TokenOverlap:
    token_count: int
    covered_count: int
    prefix_count: int
    suffix_count: int
    longest_run: int

    @property
    def coverage(self) -> float:
        if self.token_count <= 0:
            return 0.0
        return self.covered_count / self.token_count


MIN_OVERLAP_TOKENS = 4
DROP_OVERLAP_COVERAGE = 0.86
BOUNDARY_CONTEXT_SECONDS = 60.0
BOUNDARY_CONTEXT_SEGMENTS = 40
VAD_SILENCE_SEARCH_RADIUS_CAP_SECONDS = 900.0


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
    model_path: Path,
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
        "model": model_path.name,
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
    model_path: Path,
    language: str,
    beam_size: int,
    best_of: int,
    vad_threshold: float,
    vad_max_speech_duration_s: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    chunk_seconds: int,
    chunk_overlap_seconds: int,
    stitch_method: str | None,
    stitch_methods: list[str] | None = None,
    repetition_guard: bool,
    set_progress: ProgressCallback,
    log_stitch: StitchLogCallback | None = None,
    vad_cut_threshold: float | None = None,
) -> dict[str, Any]:
    audio_duration_seconds = probe_duration_seconds(input_path)
    chunk_seconds = max(int(chunk_seconds), 0)
    chunk_overlap_seconds = max(int(chunk_overlap_seconds), 0)
    if chunk_seconds > 0 and chunk_overlap_seconds >= chunk_seconds:
        chunk_overlap_seconds = max(chunk_seconds - 1, 0)
    stitch_method = stitch_utils.normalize_stitch_method(stitch_method, settings.stitch_method)

    if chunk_seconds > 0 and audio_duration_seconds > chunk_seconds:
        return _run_chunked_transcription(
            job_id=job_id,
            job_dir=job_dir,
            input_path=input_path,
            settings=settings,
            model_path=model_path,
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            vad_threshold=vad_threshold,
            vad_cut_threshold=vad_cut_threshold,
            vad_max_speech_duration_s=vad_max_speech_duration_s,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
            chunk_seconds=chunk_seconds,
            chunk_overlap_seconds=chunk_overlap_seconds,
            stitch_method=stitch_method,
            stitch_methods=stitch_methods,
            repetition_guard=repetition_guard,
            audio_duration_seconds=audio_duration_seconds,
            set_progress=set_progress,
            log_stitch=log_stitch,
        )

    return _run_single_transcription(
        job_id=job_id,
        job_dir=job_dir,
        input_path=input_path,
        settings=settings,
        model_path=model_path,
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
    model_path: Path,
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
        model_path=model_path,
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
                model_path=model_path,
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
        model_path=model_path,
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
    result["decode"]["chunking"] = {
        "enabled": False,
        "chunk_seconds": 0,
        "overlap_seconds": 0,
        "stitch_method": None,
        "stitch_methods": None,
    }
    result_path = job_dir / "result.json"
    _write_json(result_path, result)
    return result


def _run_chunked_transcription(
    *,
    job_id: str,
    job_dir: Path,
    input_path: Path,
    settings: Settings,
    model_path: Path,
    language: str,
    beam_size: int,
    best_of: int,
    vad_threshold: float,
    vad_max_speech_duration_s: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    chunk_seconds: int,
    chunk_overlap_seconds: int,
    stitch_method: str,
    stitch_methods: list[str] | None,
    repetition_guard: bool,
    audio_duration_seconds: float,
    set_progress: ProgressCallback,
    log_stitch: StitchLogCallback | None,
    vad_cut_threshold: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = job_dir / "whisper_stdout.log"
    stderr_log = job_dir / "whisper_stderr.log"
    primary_stitch_method = stitch_utils.normalize_stitch_method(stitch_method, settings.stitch_method)
    requested_stitch_methods = normalize_requested_stitch_methods(
        primary_stitch_method,
        stitch_methods,
        settings.stitch_method,
    )
    chunk_plan = build_chunk_window_plan(
        input_path=input_path,
        settings=settings,
        audio_duration_seconds=audio_duration_seconds,
        chunk_seconds=chunk_seconds,
        chunk_overlap_seconds=chunk_overlap_seconds,
        vad_threshold=vad_threshold,
        vad_cut_threshold=vad_cut_threshold,
        vad_max_speech_duration_s=vad_max_speech_duration_s,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
    )
    windows = chunk_plan.windows
    effective_overlap_seconds = chunk_plan.overlap_seconds
    if not windows:
        raise TranscriptionError("Unable to build transcription chunks")

    chunk_outputs: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = list(chunk_plan.warnings)
    retried_chunks: list[int] = []
    set_progress(5)
    _emit_stitch_log(
        log_stitch,
        {
            "type": "chunking_started",
            "chunk_count": len(windows),
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": effective_overlap_seconds,
            "requested_overlap_seconds": chunk_overlap_seconds,
            "strategy": chunk_plan.strategy,
            "stitch_method": primary_stitch_method,
            "stitch_methods": requested_stitch_methods,
        },
    )
    _append_text(
        stdout_log,
        (
            f"Chunked transcription enabled: {len(windows)} chunks "
            f"(strategy={chunk_plan.strategy}, overlap={effective_overlap_seconds}s).\n"
        ),
    )
    _append_text(stderr_log, "Chunked mode writes detailed whisper logs under chunks/<index>/.\n")
    for warning in chunk_plan.warnings:
        message = warning.get("message", warning.get("reason", "unknown"))
        _append_text(stderr_log, f"Chunk planning warning: {message}\n")

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
            model_path=model_path,
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

        chunk_outputs.append(
            {
                "window": window,
                "segments": chunk_segments,
                "warning": warning,
            }
        )

    set_progress(95)
    elapsed_seconds = time.monotonic() - started
    variants = build_stitch_variants(
        job_id=job_id,
        job_dir=job_dir,
        chunk_outputs=chunk_outputs,
        stitch_methods=requested_stitch_methods,
        chunk_overlap_seconds=effective_overlap_seconds,
        chunk_seconds=chunk_seconds,
        requested_overlap_seconds=chunk_overlap_seconds,
        chunking_strategy=chunk_plan.strategy,
        silence_cuts=[item.as_dict() for item in chunk_plan.cut_decisions],
        repetition_guard=repetition_guard,
        retried_chunks=retried_chunks,
        warnings=warnings,
        log_stitch=log_stitch,
        log_primary_method=primary_stitch_method,
    )
    primary_variant = variants[primary_stitch_method]
    all_segments = primary_variant["segments"]
    text = primary_variant["text"]
    rtf = elapsed_seconds / audio_duration_seconds if audio_duration_seconds > 0 else 0.0
    speedup = audio_duration_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
    result = {
        "job_id": job_id,
        "engine": "whisper.cpp",
        "model": model_path.name,
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
                "chunk_seconds": chunk_seconds,
                "overlap_seconds": effective_overlap_seconds,
                "requested_overlap_seconds": chunk_overlap_seconds,
                "strategy": chunk_plan.strategy,
                "silence_cuts": [item.as_dict() for item in chunk_plan.cut_decisions],
                "chunk_count": len(windows),
                "stitch_method": primary_stitch_method,
                "stitch_methods": requested_stitch_methods,
                "repetition_guard": repetition_guard,
                "retried_chunks": retried_chunks,
                "warning_count": len(warnings),
                "stitch_debug_path": primary_variant["decode"]["chunking"].get("stitch_debug_path"),
                "stitch_debug_json_path": primary_variant["decode"]["chunking"].get("stitch_debug_json_path"),
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
    result["stitch_variants"] = variants
    _emit_stitch_log(
        log_stitch,
        {
            "type": "chunking_finished",
            "segment_count": len(all_segments),
            "stitch_debug_json_path": primary_variant["decode"]["chunking"].get("stitch_debug_json_path"),
        },
    )
    result_path = job_dir / "result.json"
    _write_json(result_path, result)
    return result


def normalize_requested_stitch_methods(
    stitch_method: str,
    stitch_methods: list[str] | None,
    default: str,
) -> list[str]:
    primary_method = stitch_utils.normalize_stitch_method(stitch_method, default)
    requested = stitch_methods or [primary_method]
    methods: list[str] = []
    for item in requested:
        method = stitch_utils.normalize_stitch_method(item, "")
        if method and method not in methods:
            methods.append(method)
    if primary_method not in methods:
        methods.insert(0, primary_method)
    return methods


def build_stitch_variants(
    *,
    job_id: str,
    job_dir: Path,
    chunk_outputs: list[dict[str, Any]],
    stitch_methods: list[str],
    chunk_overlap_seconds: int,
    chunk_seconds: int,
    repetition_guard: bool,
    retried_chunks: list[int],
    warnings: list[dict[str, Any]],
    log_stitch: StitchLogCallback | None,
    log_primary_method: str,
    requested_overlap_seconds: int | None = None,
    chunking_strategy: str = "fixed",
    silence_cuts: list[dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for method in stitch_methods:
        debug_path = job_dir / ("stitch_debug.md" if method == log_primary_method else f"stitch_debug_{method}.md")
        debug_json_path = job_dir / (
            "stitch_debug.json" if method == log_primary_method else f"stitch_debug_{method}.json"
        )
        _write_stitch_debug_header(debug_path, job_id=job_id)

        all_segments: list[dict[str, Any]] = []
        stitch_audits: list[dict[str, Any]] = []
        for item in chunk_outputs:
            window = item["window"]
            chunk_segments = item["segments"]
            warning = item["warning"]
            boundary_overlap_seconds = window.previous_overlap_seconds if window.index > 0 else 0
            all_segments, audit = stitch_utils.merge_chunk_segments_with_audit(
                all_segments,
                chunk_segments,
                previous_chunk_index=window.index - 1,
                next_chunk_index=window.index,
                overlap_start_seconds=window.start_seconds,
                overlap_end_seconds=window.start_seconds + boundary_overlap_seconds if window.index > 0 else None,
                overlap_seconds=boundary_overlap_seconds,
                leading_overlap_seconds=boundary_overlap_seconds,
                trailing_overlap_seconds=window.next_overlap_seconds,
                chunk_start_seconds=window.start_seconds,
                chunk_end_seconds=window.start_seconds + window.duration_seconds,
                is_first_chunk=window.index == 0,
                is_last_chunk=window.index == len(chunk_outputs) - 1,
                incoming_warning=warning,
                stitch_method=method,
            )
            if audit is None:
                continue
            stitch_audits.append(audit)
            _append_text(debug_path, stitch_utils.render_stitch_audit_markdown(audit))
            if method == log_primary_method:
                _emit_stitch_log(
                    log_stitch,
                    {
                        "type": "boundary",
                        "previous_chunk": audit.get("previous_chunk"),
                        "next_chunk": audit.get("next_chunk"),
                        "method": audit.get("method", method),
                        "overlap_start_label": audit.get("overlap_start_label"),
                        "overlap_end_label": audit.get("overlap_end_label"),
                        "counts": audit.get("counts", {}),
                    },
                )

        all_segments.sort(key=lambda segment: (segment["start"], segment["end"]))
        text = " ".join(item["transcript"] for item in all_segments if item["transcript"]).strip()
        _write_json(
            debug_json_path,
            {
                "job_id": job_id,
                "chunk_count": len(chunk_outputs),
                "overlap_seconds": chunk_overlap_seconds,
                "requested_overlap_seconds": requested_overlap_seconds,
                "strategy": chunking_strategy,
                "silence_cuts": silence_cuts or [],
                "stitch_method": method,
                "boundaries": stitch_audits,
            },
        )
        variants[method] = {
            "text": text,
            "segments": all_segments,
            "warnings": warnings,
            "decode": {
                "chunking": {
                    "enabled": True,
                    "chunk_seconds": chunk_seconds,
                    "overlap_seconds": chunk_overlap_seconds,
                    "requested_overlap_seconds": requested_overlap_seconds,
                    "strategy": chunking_strategy,
                    "silence_cuts": silence_cuts or [],
                    "chunk_count": len(chunk_outputs),
                    "stitch_method": method,
                    "repetition_guard": repetition_guard,
                    "retried_chunks": retried_chunks,
                    "warning_count": len(warnings),
                    "stitch_debug_path": str(debug_path),
                    "stitch_debug_json_path": str(debug_json_path),
                },
            },
        }

    return variants


def _run_chunk_window(
    *,
    chunk_dir: Path,
    input_path: Path,
    settings: Settings,
    model_path: Path,
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
        model_path=model_path,
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
        model_path=model_path,
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
    model_path: Path,
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
        str(model_path),
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
        windows.append(
            ChunkWindow(
                index=index,
                start_seconds=round(start, 3),
                duration_seconds=round(duration, 3),
                previous_overlap_seconds=overlap_seconds if index > 0 else 0,
                next_overlap_seconds=overlap_seconds,
            )
        )
        if start + duration >= audio_duration_seconds:
            break
        start += step
        index += 1
    return windows


def build_chunk_window_plan(
    *,
    input_path: Path,
    settings: Settings,
    audio_duration_seconds: float,
    chunk_seconds: int,
    chunk_overlap_seconds: int,
    vad_threshold: float,
    vad_max_speech_duration_s: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    vad_cut_threshold: float | None = None,
) -> ChunkWindowPlan:
    fixed_windows = build_chunk_windows(
        audio_duration_seconds=audio_duration_seconds,
        chunk_seconds=chunk_seconds,
        overlap_seconds=chunk_overlap_seconds,
    )
    if len(fixed_windows) <= 1:
        return ChunkWindowPlan(
            strategy="fixed",
            windows=fixed_windows,
            overlap_seconds=chunk_overlap_seconds,
            warnings=[],
            cut_decisions=[],
        )

    try:
        vad_output = run_vad_speech_segments(
            input_path=input_path,
            settings=settings,
            vad_threshold=vad_threshold if vad_cut_threshold is None else vad_cut_threshold,
            vad_max_speech_duration_s=vad_max_speech_duration_s,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
            vad_speech_pad_ms=vad_speech_pad_ms,
        )
        speech_segments = parse_vad_speech_segments(vad_output)
        windows, decisions = build_mixed_aligned_chunk_windows(
            audio_duration_seconds=audio_duration_seconds,
            chunk_seconds=chunk_seconds,
            speech_segments=speech_segments,
            silence_min_duration_ms=vad_min_silence_duration_ms,
            overlap_seconds=chunk_overlap_seconds,
        )
    except (OSError, ValueError) as exc:
        return _fixed_chunk_fallback_plan(
            fixed_windows=fixed_windows,
            overlap_seconds=chunk_overlap_seconds,
            reason=str(exc),
        )

    strategy = "vad_silence"
    effective_overlap = chunk_overlap_seconds
    if any(item.cut_type == "hard_fallback" for item in decisions):
        strategy = "mixed"

    return ChunkWindowPlan(
        strategy=strategy,
        windows=windows,
        overlap_seconds=effective_overlap,
        warnings=[],
        cut_decisions=decisions,
    )


def run_vad_speech_segments(
    *,
    input_path: Path,
    settings: Settings,
    vad_threshold: float,
    vad_max_speech_duration_s: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
) -> str:
    vad_bin = settings.whispercpp_base_dir / "build" / "bin" / "whisper-vad-speech-segments"
    if not vad_bin.exists() or not vad_bin.is_file() or vad_bin.stat().st_mode & 0o111 == 0:
        raise OSError(f"VAD speech segment helper is not executable: {vad_bin}")
    command = [
        str(vad_bin),
        "--file",
        str(input_path),
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
    ]
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise OSError(detail or f"VAD speech segment helper exited with code {proc.returncode}")
    return "\n".join(part for part in (proc.stdout, proc.stderr) if part)


def parse_vad_speech_segments(output: str) -> list[VadSpeechSegment]:
    vad_log_re = re.compile(
        r"VAD segment\s+\d+:\s+start\s*=\s*([0-9]+(?:\.[0-9]+)?),\s+end\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )
    segments = _parse_vad_segment_matches(vad_log_re.finditer(output), scale=1.0)
    if segments:
        return segments

    speech_segment_re = re.compile(
        r"Speech segment\s+\d+:\s+start\s*=\s*([0-9]+(?:\.[0-9]+)?),\s+end\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )
    return _parse_vad_segment_matches(speech_segment_re.finditer(output), scale=0.01)


def _parse_vad_segment_matches(matches: Any, *, scale: float) -> list[VadSpeechSegment]:
    segments: list[VadSpeechSegment] = []
    for match in matches:
        start = float(match.group(1)) * scale
        end = float(match.group(2)) * scale
        if end <= start:
            continue
        segments.append(VadSpeechSegment(start_seconds=round(start, 3), end_seconds=round(end, 3)))
    return sorted(segments, key=lambda item: (item.start_seconds, item.end_seconds))


def build_silence_aligned_chunk_windows(
    *,
    audio_duration_seconds: float,
    chunk_seconds: int,
    speech_segments: list[VadSpeechSegment],
    silence_min_duration_ms: int,
) -> tuple[list[ChunkWindow], list[SilenceCutDecision]]:
    if audio_duration_seconds <= 0:
        return [], []
    chunk_seconds = max(int(chunk_seconds), 1)
    min_silence_seconds = max(float(silence_min_duration_ms) / 1000.0, 0.0)
    min_chunk_duration = chunk_seconds / 2.0
    search_radius = min(chunk_seconds / 2.0, VAD_SILENCE_SEARCH_RADIUS_CAP_SECONDS)
    silences = _infer_silence_gaps(
        speech_segments=speech_segments,
        audio_duration_seconds=audio_duration_seconds,
        min_silence_seconds=min_silence_seconds,
    )
    if not silences:
        raise ValueError("VAD did not find any silence gap long enough for aligned cuts")

    decisions: list[SilenceCutDecision] = []
    cuts: list[float] = []
    previous_cut = 0.0
    target = float(chunk_seconds)
    while target < audio_duration_seconds:
        decision = _select_silence_cut(
            target_seconds=target,
            search_radius_seconds=search_radius,
            silences=silences,
        )
        if decision is None:
            raise ValueError(f"No usable silence found near target cut {target:.3f}s")
        if decision.selected_seconds - previous_cut < min_chunk_duration:
            raise ValueError(
                f"Silence cut {decision.selected_seconds:.3f}s near target {target:.3f}s "
                f"would create a chunk shorter than {min_chunk_duration:.3f}s"
            )
        cuts.append(decision.selected_seconds)
        decisions.append(decision)
        previous_cut = decision.selected_seconds
        target += chunk_seconds

    windows = _windows_from_cuts(audio_duration_seconds=audio_duration_seconds, cuts=cuts)
    if not windows:
        raise ValueError("Silence-aligned cuts did not produce any chunk windows")
    return windows, decisions


def build_mixed_aligned_chunk_windows(
    *,
    audio_duration_seconds: float,
    chunk_seconds: int,
    speech_segments: list[VadSpeechSegment],
    silence_min_duration_ms: int,
    overlap_seconds: int,
) -> tuple[list[ChunkWindow], list[SilenceCutDecision]]:
    if audio_duration_seconds <= 0:
        return [], []
    chunk_seconds = max(int(chunk_seconds), 1)
    min_silence_seconds = max(float(silence_min_duration_ms) / 1000.0, 0.0)
    min_chunk_duration = chunk_seconds / 2.0
    search_radius = min(chunk_seconds / 2.0, VAD_SILENCE_SEARCH_RADIUS_CAP_SECONDS)
    silences = _infer_silence_gaps(
        speech_segments=speech_segments,
        audio_duration_seconds=audio_duration_seconds,
        min_silence_seconds=min_silence_seconds,
    )

    decisions: list[SilenceCutDecision] = []
    previous_cut = 0.0
    target = float(chunk_seconds)
    while target < audio_duration_seconds:
        if audio_duration_seconds - target < min_chunk_duration:
            break

        decision = _select_silence_cut(
            target_seconds=target,
            search_radius_seconds=search_radius,
            silences=silences,
        )
        if (
            decision is not None
            and decision.selected_seconds - previous_cut >= min_chunk_duration
            and audio_duration_seconds - decision.selected_seconds >= min_chunk_duration
        ):
            decisions.append(decision)
            previous_cut = decision.selected_seconds
            target += chunk_seconds
            continue

        reason = "No usable silence found near target cut"
        if decision is not None and decision.selected_seconds - previous_cut < min_chunk_duration:
            reason = "Selected silence would create a short previous chunk"
        elif decision is not None and audio_duration_seconds - decision.selected_seconds < min_chunk_duration:
            reason = "Selected silence would create a short final chunk"

        if target - previous_cut >= min_chunk_duration:
            decisions.append(
                SilenceCutDecision(
                    target_seconds=target,
                    selected_seconds=round(target, 3),
                    silence_start_seconds=round(target, 3),
                    silence_end_seconds=round(target, 3),
                    cut_type="hard_fallback",
                    reason=reason,
                )
            )
            previous_cut = target
        target += chunk_seconds

    windows = _windows_from_cut_decisions(
        audio_duration_seconds=audio_duration_seconds,
        decisions=decisions,
        overlap_seconds=overlap_seconds,
    )
    if not windows:
        raise ValueError("Mixed chunk planning did not produce any chunk windows")
    return windows, decisions


def _infer_silence_gaps(
    *,
    speech_segments: list[VadSpeechSegment],
    audio_duration_seconds: float,
    min_silence_seconds: float,
) -> list[tuple[float, float]]:
    silences: list[tuple[float, float]] = []
    cursor = 0.0
    for segment in sorted(speech_segments, key=lambda item: (item.start_seconds, item.end_seconds)):
        start = max(segment.start_seconds, 0.0)
        end = min(segment.end_seconds, audio_duration_seconds)
        if start - cursor >= min_silence_seconds:
            silences.append((round(cursor, 3), round(start, 3)))
        cursor = max(cursor, end)
    if audio_duration_seconds - cursor >= min_silence_seconds:
        silences.append((round(cursor, 3), round(audio_duration_seconds, 3)))
    return silences


def _select_silence_cut(
    *,
    target_seconds: float,
    search_radius_seconds: float,
    silences: list[tuple[float, float]],
) -> SilenceCutDecision | None:
    search_start = target_seconds - search_radius_seconds
    search_end = target_seconds + search_radius_seconds
    candidates: list[SilenceCutDecision] = []
    for silence_start, silence_end in silences:
        if silence_end < search_start or silence_start > search_end:
            continue
        if silence_start <= target_seconds <= silence_end:
            selected = target_seconds
        else:
            selected = (silence_start + silence_end) / 2.0
        if selected < search_start or selected > search_end:
            continue
        candidates.append(
            SilenceCutDecision(
                target_seconds=target_seconds,
                selected_seconds=round(selected, 3),
                silence_start_seconds=silence_start,
                silence_end_seconds=silence_end,
            )
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.distance_seconds, item.selected_seconds))


def _windows_from_cuts(*, audio_duration_seconds: float, cuts: list[float]) -> list[ChunkWindow]:
    boundaries = [0.0]
    for cut in cuts:
        rounded = round(cut, 3)
        if rounded <= boundaries[-1] or rounded >= audio_duration_seconds:
            continue
        boundaries.append(rounded)
    boundaries.append(round(audio_duration_seconds, 3))
    windows: list[ChunkWindow] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        duration = round(end - start, 3)
        if duration <= 0:
            continue
        windows.append(ChunkWindow(index=index, start_seconds=round(start, 3), duration_seconds=duration))
    return windows


def _windows_from_cut_decisions(
    *,
    audio_duration_seconds: float,
    decisions: list[SilenceCutDecision],
    overlap_seconds: int,
) -> list[ChunkWindow]:
    boundaries = [0.0]
    for decision in decisions:
        rounded = round(decision.selected_seconds, 3)
        if rounded <= boundaries[-1] or rounded >= audio_duration_seconds:
            continue
        boundaries.append(rounded)
    boundaries.append(round(audio_duration_seconds, 3))

    windows: list[ChunkWindow] = []
    overlap_seconds = max(int(overlap_seconds), 0)
    for index, (boundary_start, boundary_end) in enumerate(zip(boundaries, boundaries[1:])):
        previous_decision = decisions[index - 1] if index > 0 and index - 1 < len(decisions) else None
        next_decision = decisions[index] if index < len(decisions) else None
        requested_overlap = overlap_seconds if previous_decision else 0
        next_overlap = overlap_seconds if next_decision else 0
        start = max(boundary_start - requested_overlap, 0.0)
        duration = round(boundary_end - start, 3)
        if duration <= 0:
            continue
        windows.append(
            ChunkWindow(
                index=index,
                start_seconds=round(start, 3),
                duration_seconds=duration,
                previous_overlap_seconds=int(round(boundary_start - start)),
                next_overlap_seconds=next_overlap,
            )
        )
    return windows


def _fixed_chunk_fallback_plan(
    *,
    fixed_windows: list[ChunkWindow],
    overlap_seconds: int,
    reason: str,
) -> ChunkWindowPlan:
    return ChunkWindowPlan(
        strategy="fixed_fallback",
        windows=fixed_windows,
        overlap_seconds=overlap_seconds,
        warnings=[
            {
                "type": "chunking_fixed_fallback",
                "reason": reason,
                "message": f"Silence-aligned chunk planning failed; using fixed hard cuts. Reason: {reason}",
            }
        ],
        cut_decisions=[],
    )


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

    merged = list(existing)
    incoming_decisions: list[dict[str, Any]] = []
    dropped_segments: list[dict[str, Any]] = []
    trimmed_segments: list[dict[str, Any]] = []
    kept_segments: list[dict[str, Any]] = []
    kept_overlap_segments: list[dict[str, Any]] = []
    kept_outside_overlap_segments: list[dict[str, Any]] = []
    dropped_exact_count = 0
    dropped_overlap_count = 0
    trimmed_prefix_count = 0
    trimmed_suffix_count = 0
    kept_low_confidence_count = 0
    for segment in incoming:
        segment_text = _normalize_for_compare(segment["transcript"])
        overlap_relevant = segment["start"] < overlap_end_seconds and segment["end"] > overlap_start_seconds
        context_segments = _boundary_context_segments(merged, overlap_start_seconds)
        recent_texts = {
            _normalize_for_compare(item["transcript"])
            for item in context_segments
            if _normalize_for_compare(item["transcript"])
        }

        if overlap_relevant and segment_text and segment_text in recent_texts:
            dropped_segments.append(_audit_segment(segment))
            dropped_exact_count += 1
            incoming_decisions.append(
                {
                    "decision": "dropped_exact_duplicate",
                    "segment": _audit_segment(segment),
                }
            )
            continue

        if overlap_relevant:
            context_tokens = _comparison_context_tokens(context_segments)
            segment_tokens = _tokenize_for_compare(segment["transcript"])
            token_overlap = _measure_token_overlap(context_tokens, segment_tokens)
            if _should_drop_overlap_duplicate(token_overlap):
                dropped_segments.append(_audit_segment(segment))
                dropped_overlap_count += 1
                incoming_decisions.append(
                    {
                        "decision": "dropped_overlap_duplicate",
                        "segment": _audit_segment(segment),
                        "overlap": _audit_token_overlap(token_overlap),
                    }
                )
                continue

            trimmed_segment: dict[str, Any] | None = None
            decision = "kept_low_confidence_overlap"
            if _should_trim_duplicate_prefix(token_overlap):
                trimmed_segment = _trim_segment_tokens(segment, trim_prefix_tokens=token_overlap.prefix_count)
                decision = "trimmed_duplicate_prefix"
                trimmed_prefix_count += 1
            elif _should_trim_duplicate_suffix(token_overlap):
                trimmed_segment = _trim_segment_tokens(segment, trim_suffix_tokens=token_overlap.suffix_count)
                decision = "trimmed_duplicate_suffix"
                trimmed_suffix_count += 1

            if trimmed_segment is not None:
                audited = _audit_segment(trimmed_segment)
                trimmed_segments.append(audited)
                kept_segments.append(audited)
                kept_overlap_segments.append(audited)
                incoming_decisions.append(
                    {
                        "decision": decision,
                        "segment": audited,
                        "original_segment": _audit_segment(segment),
                        "overlap": _audit_token_overlap(token_overlap),
                    }
                )
                merged.append(trimmed_segment)
                continue

            kept_low_confidence_count += 1
        else:
            decision = "kept_outside_overlap"

        audited = _audit_segment(segment)
        kept_segments.append(audited)
        if overlap_relevant:
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
        "trimmed_duplicates": trimmed_segments,
        "kept_overlap": kept_overlap_segments,
        "kept_outside_overlap_head": kept_outside_overlap_segments[:8],
        "incoming_warning": incoming_warning,
        "counts": {
            "incoming": len(incoming),
            "dropped_duplicates": len(dropped_segments),
            "dropped_exact_duplicates": dropped_exact_count,
            "dropped_overlap_duplicates": dropped_overlap_count,
            "trimmed_duplicate_prefix": trimmed_prefix_count,
            "trimmed_duplicate_suffix": trimmed_suffix_count,
            "kept": len(kept_segments),
            "kept_overlap": len(kept_overlap_segments),
            "kept_outside_overlap": len(kept_outside_overlap_segments),
            "kept_low_confidence_overlap": kept_low_confidence_count,
        },
    }
    return merged, audit


def _boundary_context_segments(
    segments: list[dict[str, Any]],
    overlap_start_seconds: float,
) -> list[dict[str, Any]]:
    context_start = overlap_start_seconds - BOUNDARY_CONTEXT_SECONDS
    nearby_segments = [item for item in segments if item["end"] >= context_start]
    return nearby_segments[-BOUNDARY_CONTEXT_SEGMENTS:]


def _comparison_context_tokens(segments: list[dict[str, Any]]) -> list[str]:
    return _tokenize_for_compare(" ".join(item.get("transcript", "") for item in segments))


def _tokenize_for_compare(text: str) -> list[str]:
    normalized = _normalize_for_compare(text)
    if not normalized:
        return []
    return normalized.split()


def _measure_token_overlap(context_tokens: list[str], segment_tokens: list[str]) -> TokenOverlap:
    if not context_tokens or not segment_tokens:
        return TokenOverlap(
            token_count=len(segment_tokens),
            covered_count=0,
            prefix_count=0,
            suffix_count=0,
            longest_run=0,
        )

    covered = [False] * len(segment_tokens)
    longest_run = 0
    matcher = SequenceMatcher(None, context_tokens, segment_tokens, autojunk=False)
    for _context_start, segment_start, size in matcher.get_matching_blocks():
        if size <= 0:
            continue
        longest_run = max(longest_run, size)
        for index in range(segment_start, segment_start + size):
            covered[index] = True

    prefix_count = 0
    for is_covered in covered:
        if not is_covered:
            break
        prefix_count += 1

    suffix_count = 0
    for is_covered in reversed(covered):
        if not is_covered:
            break
        suffix_count += 1

    return TokenOverlap(
        token_count=len(segment_tokens),
        covered_count=sum(1 for is_covered in covered if is_covered),
        prefix_count=prefix_count,
        suffix_count=suffix_count,
        longest_run=longest_run,
    )


def _should_drop_overlap_duplicate(token_overlap: TokenOverlap) -> bool:
    if token_overlap.token_count < MIN_OVERLAP_TOKENS:
        return False
    return (
        token_overlap.coverage >= DROP_OVERLAP_COVERAGE
        and token_overlap.longest_run >= min(MIN_OVERLAP_TOKENS, token_overlap.token_count)
    )


def _should_trim_duplicate_prefix(token_overlap: TokenOverlap) -> bool:
    if token_overlap.token_count <= MIN_OVERLAP_TOKENS:
        return False
    if token_overlap.prefix_count < MIN_OVERLAP_TOKENS:
        return False
    return token_overlap.prefix_count < token_overlap.token_count


def _should_trim_duplicate_suffix(token_overlap: TokenOverlap) -> bool:
    if token_overlap.token_count <= MIN_OVERLAP_TOKENS:
        return False
    if token_overlap.suffix_count < MIN_OVERLAP_TOKENS:
        return False
    return token_overlap.suffix_count < token_overlap.token_count


def _trim_segment_tokens(
    segment: dict[str, Any],
    *,
    trim_prefix_tokens: int = 0,
    trim_suffix_tokens: int = 0,
) -> dict[str, Any] | None:
    token_spans = _token_spans(segment["transcript"])
    token_count = len(token_spans)
    keep_start_token = min(max(trim_prefix_tokens, 0), token_count)
    keep_end_token = max(min(token_count - max(trim_suffix_tokens, 0), token_count), 0)
    if token_count == 0 or keep_start_token >= keep_end_token:
        return None

    text_start = token_spans[keep_start_token][1]
    text_end = token_spans[keep_end_token - 1][2]
    transcript = segment["transcript"][text_start:text_end].strip()
    if not transcript:
        return None

    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(end - start, 0.0)
    trimmed_start = start + duration * (keep_start_token / token_count)
    trimmed_end = start + duration * (keep_end_token / token_count)

    trimmed = dict(segment)
    trimmed["start"] = round(trimmed_start, 3)
    trimmed["end"] = round(max(trimmed_start, trimmed_end), 3)
    trimmed["transcript"] = transcript
    trimmed["words"] = _trim_words(
        segment.get("words") or [],
        keep_start_token=keep_start_token,
        keep_end_token=keep_end_token,
        start=trimmed["start"],
        end=trimmed["end"],
    )
    return trimmed


def _trim_words(
    words: list[dict[str, Any]],
    *,
    keep_start_token: int,
    keep_end_token: int,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for word in words[keep_start_token:keep_end_token]:
        clipped = _clip_word(word, start=start, end=end)
        if clipped is not None:
            trimmed.append(clipped)
    return trimmed


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).lower(), match.start(), match.end())
        for match in re.finditer(r"[\w']+", text, flags=re.UNICODE)
    ]


def _audit_token_overlap(token_overlap: TokenOverlap) -> dict[str, Any]:
    return {
        "token_count": token_overlap.token_count,
        "covered_count": token_overlap.covered_count,
        "coverage": round(token_overlap.coverage, 3),
        "prefix_count": token_overlap.prefix_count,
        "suffix_count": token_overlap.suffix_count,
        "longest_run": token_overlap.longest_run,
    }


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
        original_segment = item.get("original_segment")
        if original_segment:
            lines.append(
                (
                    f"- {item['decision']} "
                    f"[{original_segment['start_label']} -> {original_segment['end_label']}] "
                    f"=> [{segment['start_label']} -> {segment['end_label']}] {segment['transcript']}"
                )
            )
        else:
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
                f"Dropped {counts.get('dropped_duplicates', 0)} duplicate segment(s) from overlap "
                f"({counts.get('dropped_exact_duplicates', 0)} exact, "
                f"{counts.get('dropped_overlap_duplicates', 0)} token-overlap), "
                f"trimmed {counts.get('trimmed_duplicate_prefix', 0)} prefix and "
                f"{counts.get('trimmed_duplicate_suffix', 0)} suffix duplicate segment(s), "
                f"kept {counts.get('kept', 0)} incoming segment(s) "
                f"({counts.get('kept_overlap', 0)} inside overlap, "
                f"{counts.get('kept_outside_overlap', 0)} outside overlap, "
                f"{counts.get('kept_low_confidence_overlap', 0)} low-confidence overlap)."
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
    token_time_origin: float | None = None
    previous_token_end_raw: float | None = None
    for item in source_segments:
        segment_text = (item.get("text") or item.get("transcript") or "").strip()
        raw_start, raw_end = _extract_segment_times(item)
        start = raw_start + offset_seconds
        end = raw_end + offset_seconds
        segment_time_origin = start
        current_token_time_origin = None
        token_start_raw, token_end_raw = _token_time_bounds(item)
        if token_start_raw is not None and token_end_raw is not None:
            token_looks_relative = _token_bounds_look_relative_to_segment(
                token_start_raw=token_start_raw,
                token_end_raw=token_end_raw,
                segment_start_raw=raw_start,
                segment_end_raw=raw_end,
            )
            if token_looks_relative:
                token_reset = previous_token_end_raw is not None and token_start_raw < previous_token_end_raw - 1.0
                if token_time_origin is None or token_reset:
                    token_time_origin = segment_time_origin
                current_token_time_origin = token_time_origin
                previous_token_end_raw = token_end_raw
            elif _token_bounds_look_vad_shifted(
                token_start_raw=token_start_raw,
                token_end_raw=token_end_raw,
                segment_start_raw=raw_start,
                segment_end_raw=raw_end,
            ):
                current_token_time_origin = segment_time_origin - token_start_raw
                token_time_origin = None
                previous_token_end_raw = None
            else:
                token_time_origin = None
                previous_token_end_raw = None
        if clamp_start is not None:
            start = max(start, clamp_start)
        if clamp_end is not None:
            end = min(end, clamp_end)
        if end < start:
            end = start
        words = _extract_words(
            item,
            offset_seconds=offset_seconds,
            token_time_origin=current_token_time_origin,
            clip_start=clamp_start,
            clip_end=clamp_end,
        )
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "transcript": segment_text,
                "words": words,
            }
        )
    return segments


def _extract_words(
    segment: dict[str, Any],
    *,
    offset_seconds: float,
    token_time_origin: float | None,
    clip_start: float | None,
    clip_end: float | None,
) -> list[dict[str, Any]]:
    tokens = segment.get("tokens")
    if not isinstance(tokens, list):
        return []

    words: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_start = 0.0
    current_end = 0.0

    def flush() -> None:
        nonlocal current_parts, current_start, current_end
        if not current_parts:
            return
        text = "".join(current_parts).strip()
        current_parts = []
        if not text:
            return
        word = _clip_word(
            {
                "word": text,
                "start": round(current_start, 3),
                "end": round(max(current_start, current_end), 3),
            },
            start=clip_start,
            end=clip_end,
        )
        if word is not None:
            words.append(word)

    for token in tokens:
        if not isinstance(token, dict):
            continue
        raw_text = token.get("text")
        if not isinstance(raw_text, str):
            continue
        if _is_special_token(raw_text):
            continue

        piece = raw_text.strip()
        if not piece:
            continue
        has_word_char = re.search(r"\w", piece, flags=re.UNICODE) is not None
        is_apostrophe = piece in {"'", "’"}
        if not has_word_char and not is_apostrophe:
            continue
        if is_apostrophe and not current_parts:
            continue

        token_start_raw, token_end_raw = _extract_segment_times(token)
        if token_time_origin is None:
            token_start = token_start_raw + offset_seconds
            token_end = token_end_raw + offset_seconds
        else:
            token_start = token_start_raw + token_time_origin
            token_end = token_end_raw + token_time_origin
        if token_end < token_start:
            token_end = token_start

        starts_new_word = raw_text[0].isspace() and bool(current_parts) and has_word_char
        if starts_new_word:
            flush()

        if not current_parts:
            current_start = token_start
            current_end = token_end
        else:
            current_start = min(current_start, token_start)
            current_end = max(current_end, token_end)
        current_parts.append(piece)

    flush()
    return words


def _token_bounds_look_relative_to_segment(
    *,
    token_start_raw: float,
    token_end_raw: float,
    segment_start_raw: float,
    segment_end_raw: float,
) -> bool:
    if token_start_raw >= segment_start_raw - 1.0:
        return False

    segment_duration = max(segment_end_raw - segment_start_raw, 0.0)
    token_span_fits_segment = token_end_raw <= max(segment_duration + 5.0, 30.0)
    if not token_span_fits_segment:
        return False

    token_starts_far_before_segment = segment_start_raw - token_start_raw > 5.0
    token_ends_before_segment = segment_start_raw - token_end_raw > 1.0
    return token_starts_far_before_segment or token_ends_before_segment


def _token_bounds_look_vad_shifted(
    *,
    token_start_raw: float,
    token_end_raw: float,
    segment_start_raw: float,
    segment_end_raw: float,
) -> bool:
    if token_start_raw >= segment_start_raw:
        return False
    if segment_start_raw - token_end_raw <= 2.0:
        return False

    segment_duration = max(segment_end_raw - segment_start_raw, 0.0)
    token_duration = max(token_end_raw - token_start_raw, 0.0)
    return token_duration <= max(segment_duration + 5.0, 30.0)


def _token_time_bounds(segment: dict[str, Any]) -> tuple[float | None, float | None]:
    tokens = segment.get("tokens")
    if not isinstance(tokens, list):
        return None, None

    starts: list[float] = []
    ends: list[float] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        raw_text = token.get("text")
        if not isinstance(raw_text, str) or _is_special_token(raw_text):
            continue
        piece = raw_text.strip()
        if not piece or re.search(r"\w", piece, flags=re.UNICODE) is None:
            continue
        token_start, token_end = _extract_segment_times(token)
        starts.append(token_start)
        ends.append(max(token_start, token_end))

    if not starts:
        return None, None
    return min(starts), max(ends)


def _is_special_token(text: str) -> bool:
    stripped = text.strip()
    return (stripped.startswith("[_") and stripped.endswith("]")) or (
        stripped.startswith("<|") and stripped.endswith("|>")
    )


def _clip_word(
    word: dict[str, Any],
    *,
    start: float | None,
    end: float | None,
) -> dict[str, Any] | None:
    word_start = float(word.get("start", 0.0))
    word_end = float(word.get("end", word_start))
    if start is not None and word_end < start:
        return None
    if end is not None and word_start > end:
        return None

    clipped_start = word_start if start is None else max(word_start, start)
    clipped_end = max(word_start, word_end) if end is None else min(max(word_start, word_end), end)
    clipped_start = round(clipped_start, 3)
    clipped_end = round(clipped_end, 3)
    if clipped_end < clipped_start:
        clipped_end = clipped_start

    return {
        "word": str(word.get("word", "")),
        "start": clipped_start,
        "end": clipped_end,
    }


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


def _emit_stitch_log(callback: StitchLogCallback | None, event: dict[str, Any]) -> None:
    if callback is None:
        return
    callback(event)


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
