import json
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from .config import Settings
from .whisper import run_transcription


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.RLock()
        self.wakeup = threading.Event()
        self.stopped = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.logger = logging.getLogger("uvicorn.error")

    def initialize(self) -> None:
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._mark_interrupted_jobs_failed()

    def start_worker(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.worker_thread = threading.Thread(target=self._worker_loop, name="whispercpp-worker", daemon=True)
        self.worker_thread.start()

    def stop_worker(self) -> None:
        self.stopped.set()
        self.wakeup.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    async def create_upload_job(
        self,
        *,
        upload: UploadFile,
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
    ) -> dict[str, Any]:
        job_id, job_dir = self._new_job_dir()
        filename = Path(upload.filename or "upload").name
        source_path = job_dir / filename
        with source_path.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

        metadata = self._base_metadata(
            job_id=job_id,
            job_dir=job_dir,
            source_kind="upload",
            source_path=source_path,
            original_filename=filename,
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
        )
        self._write_metadata(job_dir, metadata)
        self.wakeup.set()
        return metadata

    def create_path_job(
        self,
        *,
        path: Path,
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
    ) -> dict[str, Any]:
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=400, detail="Path does not exist or is not a file")
        if not os.access(path, os.R_OK):
            raise HTTPException(status_code=400, detail="Path is not readable")

        job_id, job_dir = self._new_job_dir()
        metadata = self._base_metadata(
            job_id=job_id,
            job_dir=job_dir,
            source_kind="path",
            source_path=path,
            original_filename=path.name,
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
        )
        self._write_metadata(job_dir, metadata)
        self.wakeup.set()
        return metadata

    def get_job(self, job_id: str) -> dict[str, Any]:
        job_dir = self._job_dir(job_id)
        metadata_path = job_dir / "metadata.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        return self._read_metadata(job_dir)

    def list_terminal_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        with self.lock:
            for metadata_path in self.settings.jobs_dir.glob("*/metadata.json"):
                try:
                    metadata = self._read_metadata(metadata_path.parent)
                except (OSError, json.JSONDecodeError):
                    continue
                if metadata.get("status") not in {"succeeded", "failed"}:
                    continue
                jobs.append(self._job_summary(metadata))

        return sorted(
            jobs,
            key=lambda item: item.get("completed_at") or item.get("updated_at") or item.get("created_at") or "",
            reverse=True,
        )

    def get_result(self, job_id: str) -> dict[str, Any]:
        metadata = self.get_job(job_id)
        status = metadata.get("status")
        if status == "failed":
            raise HTTPException(status_code=409, detail=metadata.get("error") or "Job failed")
        if status != "succeeded":
            raise HTTPException(status_code=409, detail=f"Job is {status}")

        result_path = Path(metadata["result_path"])
        if not result_path.exists():
            raise HTTPException(status_code=500, detail="Result file is missing")
        with result_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def get_transcript_markdown(self, job_id: str) -> str:
        metadata = self.get_job(job_id)
        status = metadata.get("status")
        if status == "failed":
            raise HTTPException(status_code=409, detail=metadata.get("error") or "Job failed")
        if status != "succeeded":
            raise HTTPException(status_code=409, detail=f"Job is {status}")

        result = self.get_result(job_id)
        return render_transcript_markdown(metadata, result)

    def get_llm_prompt(self, job_id: str) -> str:
        result = self.get_result(job_id)
        return render_llm_prompt(result)

    def transcript_download_name(self, job_id: str) -> str:
        metadata = self.get_job(job_id)
        source = metadata.get("source") or {}
        stem = Path(source.get("filename") or f"job-{job_id}").stem
        safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in stem).strip("-")
        if not safe_stem:
            safe_stem = "transcript"
        return f"{safe_stem}-{job_id[:8]}.md"

    def llm_prompt_download_name(self, job_id: str) -> str:
        metadata = self.get_job(job_id)
        source = metadata.get("source") or {}
        stem = Path(source.get("filename") or f"job-{job_id}").stem
        safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in stem).strip("-")
        if not safe_stem:
            safe_stem = "transcript"
        return f"{safe_stem}-{job_id[:8]}-llm-prompt.txt"

    def delete_job(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        if not (job_dir / "metadata.json").exists():
            raise HTTPException(status_code=404, detail="Job not found")
        shutil.rmtree(job_dir)

    def _worker_loop(self) -> None:
        while not self.stopped.is_set():
            job = self._claim_next_job()
            if job is None:
                self.wakeup.wait(timeout=1)
                self.wakeup.clear()
                continue

            job_dir = self._job_dir(job["job_id"])
            try:
                params = job["params"]
                result = run_transcription(
                    job_id=job["job_id"],
                    job_dir=job_dir,
                    input_path=Path(job["source"]["path"]),
                    settings=self.settings,
                    language=params["language"],
                    beam_size=params["beam_size"],
                    best_of=params["best_of"],
                    vad_threshold=params.get("vad_threshold", 0.5),
                    vad_max_speech_duration_s=params["vad_max_speech_duration_s"],
                    vad_min_silence_duration_ms=params["vad_min_silence_duration_ms"],
                    vad_speech_pad_ms=params["vad_speech_pad_ms"],
                    chunking_mode=params.get("chunking", {}).get("mode", self.settings.chunking_mode),
                    chunk_seconds=params.get("chunking", {}).get("chunk_seconds", self.settings.chunk_seconds),
                    chunk_overlap_seconds=params.get("chunking", {}).get(
                        "overlap_seconds", self.settings.chunk_overlap_seconds
                    ),
                    repetition_guard=params.get("chunking", {}).get(
                        "repetition_guard", self.settings.repetition_guard
                    ),
                    set_progress=lambda progress: self._set_progress(job["job_id"], progress),
                )
                self._mark_succeeded(job["job_id"], result)
            except Exception as exc:
                self._mark_failed(job["job_id"], str(exc))

    def _claim_next_job(self) -> dict[str, Any] | None:
        with self.lock:
            queued = []
            for metadata_path in self.settings.jobs_dir.glob("*/metadata.json"):
                try:
                    metadata = self._read_metadata(metadata_path.parent)
                except (OSError, json.JSONDecodeError):
                    continue
                if metadata.get("status") == "queued":
                    queued.append(metadata)

            if not queued:
                return None

            job = sorted(queued, key=lambda item: item.get("created_at", ""))[0]
            job["status"] = "running"
            job["progress"] = max(int(job.get("progress") or 0), 1)
            job["started_at"] = utc_now()
            job["updated_at"] = utc_now()
            self._write_metadata(self._job_dir(job["job_id"]), job)
            self._log_progress(job["job_id"], job["progress"])
            return job

    def _set_progress(self, job_id: str, progress: int) -> None:
        with self.lock:
            job_dir = self._job_dir(job_id)
            metadata = self._read_metadata(job_dir)
            if metadata.get("status") != "running":
                return
            old_progress = int(metadata.get("progress") or 0)
            new_progress = max(old_progress, min(progress, 99))
            if new_progress == old_progress:
                return
            metadata["progress"] = new_progress
            metadata["updated_at"] = utc_now()
            self._write_metadata(job_dir, metadata)
            self._log_progress(job_id, new_progress)

    def _mark_succeeded(self, job_id: str, result: dict[str, Any]) -> None:
        with self.lock:
            job_dir = self._job_dir(job_id)
            metadata = self._read_metadata(job_dir)
            metadata["status"] = "succeeded"
            metadata["progress"] = 100
            metadata["error"] = None
            metadata["result_path"] = str(job_dir / "result.json")
            metadata["completed_at"] = utc_now()
            metadata["updated_at"] = utc_now()
            metadata["metrics"] = result.get("metrics", {})
            metadata["model"] = result.get("model")
            self._write_metadata(job_dir, metadata)
            self._log_progress(job_id, 100)

    def _mark_failed(self, job_id: str, error: str) -> None:
        with self.lock:
            job_dir = self._job_dir(job_id)
            metadata = self._read_metadata(job_dir)
            metadata["status"] = "failed"
            metadata["progress"] = min(int(metadata.get("progress") or 0), 99)
            metadata["error"] = error
            metadata["completed_at"] = utc_now()
            metadata["updated_at"] = utc_now()
            self._write_metadata(job_dir, metadata)

    def _mark_interrupted_jobs_failed(self) -> None:
        with self.lock:
            for metadata_path in self.settings.jobs_dir.glob("*/metadata.json"):
                try:
                    metadata = self._read_metadata(metadata_path.parent)
                except (OSError, json.JSONDecodeError):
                    continue
                if metadata.get("status") == "running":
                    metadata["status"] = "failed"
                    metadata["error"] = "Job was interrupted by API restart"
                    metadata["completed_at"] = utc_now()
                    metadata["updated_at"] = utc_now()
                    self._write_metadata(metadata_path.parent, metadata)

    def _base_metadata(
        self,
        *,
        job_id: str,
        job_dir: Path,
        source_kind: str,
        source_path: Path,
        original_filename: str,
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
    ) -> dict[str, Any]:
        now = utc_now()
        return {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "source": {
                "kind": source_kind,
                "path": str(source_path),
                "filename": original_filename,
                "size_bytes": source_path.stat().st_size,
            },
            "params": {
                "language": language,
                "beam_size": beam_size,
                "best_of": best_of,
                "vad_threshold": vad_threshold,
                "vad_max_speech_duration_s": vad_max_speech_duration_s,
                "vad_min_silence_duration_ms": vad_min_silence_duration_ms,
                "vad_speech_pad_ms": vad_speech_pad_ms,
                "chunking": {
                    "mode": chunking_mode,
                    "chunk_seconds": chunk_seconds,
                    "overlap_seconds": chunk_overlap_seconds,
                    "repetition_guard": repetition_guard,
                },
            },
            "logs": {
                "stdout": str(job_dir / "whisper_stdout.log"),
                "stderr": str(job_dir / "whisper_stderr.log"),
            },
            "result_path": None,
            "error": None,
        }

    def _new_job_dir(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_id, job_dir

    def _job_dir(self, job_id: str) -> Path:
        if not re_is_hex_uuid(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        return self.settings.jobs_dir / job_id

    def _read_metadata(self, job_dir: Path) -> dict[str, Any]:
        with (job_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_metadata(self, job_dir: Path, metadata: dict[str, Any]) -> None:
        tmp_path = job_dir / "metadata.json.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        tmp_path.replace(job_dir / "metadata.json")

    def _log_progress(self, job_id: str, progress: int) -> None:
        self.logger.info("Job %s progress %s%%", job_id, progress)

    def _job_summary(self, metadata: dict[str, Any]) -> dict[str, Any]:
        model = metadata.get("model")
        if not model and metadata.get("status") == "succeeded":
            model = self._read_result_model(metadata)
        source = metadata.get("source") or {}

        return {
            "job_id": metadata.get("job_id"),
            "status": metadata.get("status"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "completed_at": metadata.get("completed_at"),
            "source": {
                "kind": source.get("kind"),
                "filename": source.get("filename"),
                "size_bytes": source.get("size_bytes"),
            },
            "params": metadata.get("params") or {},
            "metrics": metadata.get("metrics") or {},
            "model": model,
            "error": metadata.get("error"),
            "has_result": bool(metadata.get("result_path") and Path(metadata["result_path"]).exists()),
        }

    def _read_result_model(self, metadata: dict[str, Any]) -> str | None:
        result_path = metadata.get("result_path")
        if not result_path:
            return None
        try:
            with Path(result_path).open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        model = result.get("model")
        return model if isinstance(model, str) and model else None


def render_transcript_markdown(metadata: dict[str, Any], result: dict[str, Any]) -> str:
    source = metadata.get("source") or {}
    params = metadata.get("params") or {}
    metrics = result.get("metrics") or metadata.get("metrics") or {}
    decode = result.get("decode") or {}
    lines = [
        f"# Transcript: {source.get('filename') or metadata.get('job_id')}",
        "",
        "## Job",
        "",
        f"- Job ID: {metadata.get('job_id')}",
        f"- Status: {metadata.get('status')}",
        f"- Source: {source.get('filename') or 'unknown'}",
        f"- Completed at: {metadata.get('completed_at') or 'unknown'}",
        f"- Engine: {result.get('engine') or 'unknown'}",
        f"- Model: {result.get('model') or metadata.get('model') or 'unknown'}",
        f"- Language: {result.get('language') or params.get('language') or 'unknown'}",
        "",
        "## Metrics",
        "",
        f"- Audio duration: {_format_seconds(metrics.get('audio_duration_seconds'))}",
        f"- Elapsed: {_format_seconds(metrics.get('elapsed_seconds'))}",
        f"- RTF: {_format_number(metrics.get('rtf'))}",
        f"- Speedup: {_format_number(metrics.get('speedup'))}",
        "",
        "## Decode",
        "",
        f"- Beam size: {decode.get('beam_size', params.get('beam_size', 'unknown'))}",
        f"- Best of: {decode.get('best_of', params.get('best_of', 'unknown'))}",
    ]

    chunking = decode.get("chunking") or params.get("chunking") or {}
    if chunking:
        lines.extend(
            [
                f"- Chunking mode: {chunking.get('mode', 'unknown')}",
                f"- Chunk seconds: {chunking.get('chunk_seconds', 'unknown')}",
                f"- Overlap seconds: {chunking.get('overlap_seconds', 'unknown')}",
                f"- Repetition guard: {chunking.get('repetition_guard', 'unknown')}",
            ]
        )

    lines.extend(["", "## Transcript", ""])
    segments = result.get("segments") or []
    if segments:
        for segment in segments:
            transcript = str(segment.get("transcript") or "").strip()
            if transcript:
                lines.append(transcript)
                lines.append("")
    else:
        text = str(result.get("text") or "").strip()
        lines.append(text if text else "_No transcript text available._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_llm_prompt(result: dict[str, Any]) -> str:
    transcript = _compact_transcript_text(result)
    return (
        "Summarize the following meeting transcript. Return a concise meeting summary, "
        "key decisions, action items with owners when mentioned, and open questions.\n\n"
        "Transcript:\n"
        f"{transcript}\n"
    )


def _compact_transcript_text(result: dict[str, Any]) -> str:
    text = str(result.get("text") or "").strip()
    if not text:
        text = "\n".join(
            str(segment.get("transcript") or "").strip()
            for segment in result.get("segments") or []
            if str(segment.get("transcript") or "").strip()
        )
    return " ".join(text.split())


def _format_seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}s"
    return "unknown"


def _format_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "unknown"


def re_is_hex_uuid(value: str) -> bool:
    if len(value) != 32:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)
