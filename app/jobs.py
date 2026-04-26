import json
import logging
import os
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


def re_is_hex_uuid(value: str) -> bool:
    if len(value) != 32:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)
