import logging
import re
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from . import stitch_utils
from .config import get_settings
from .jobs import JobStore


settings = get_settings()
job_store = JobStore(settings)
app = FastAPI(title="Whisper.cpp Wrapper API", version="0.1.0")
STITCH_METHOD_PATTERN = f"^({'|'.join(sorted(stitch_utils.STITCH_METHODS))})$"


class StatusPollAccessLogFilter(logging.Filter):
    status_poll_re = re.compile(r'"GET /jobs/[0-9a-fA-F]{32}(?:\?.*)? HTTP/')

    def filter(self, record: logging.LogRecord) -> bool:
        return self.status_poll_re.search(record.getMessage()) is None


def install_access_log_filter() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if any(isinstance(item, StatusPollAccessLogFilter) for item in access_logger.filters):
        return
    access_logger.addFilter(StatusPollAccessLogFilter())


class PathTranscriptionRequest(BaseModel):
    path: str
    model: str
    language: str | None = None
    beam_size: int | None = Field(default=None, ge=1)
    best_of: int | None = Field(default=None, ge=1)
    vad_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    vad_max_speech_duration_s: int = Field(default=3600, ge=1)
    vad_min_silence_duration_ms: int = Field(default=2000, ge=0)
    vad_speech_pad_ms: int = Field(default=400, ge=0)
    chunk_seconds: int | None = Field(default=None, ge=0)
    chunk_overlap_seconds: int | None = Field(default=None, ge=0)
    stitch_method: str | None = Field(default=None, pattern=STITCH_METHOD_PATTERN)
    stitch_methods: list[str] | None = None
    repetition_guard: bool | None = None

    @field_validator("stitch_methods")
    @classmethod
    def validate_stitch_methods(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        methods = []
        for item in value:
            method = stitch_utils.normalize_stitch_method(item, "")
            if not method:
                raise ValueError(f"Unknown stitch method: {item}")
            if method not in methods:
                methods.append(method)
        if not methods:
            raise ValueError("stitch_methods must contain at least one method")
        return methods


@app.on_event("startup")
def startup() -> None:
    install_access_log_filter()
    settings.validate_startup()
    job_store.initialize()
    job_store.start_worker()


@app.on_event("shutdown")
def shutdown() -> None:
    job_store.stop_worker()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    startup_checks = settings.startup_checks()

    checks = {
        **startup_checks,
        "temp_dir": {
            "path": str(settings.temp_dir),
            "ok": settings.temp_dir.exists() and settings.temp_dir.is_dir(),
        },
        "ffmpeg": _binary_check("ffmpeg"),
        "ffprobe": _binary_check("ffprobe"),
    }
    return {
        "ok": all(item["ok"] for item in checks.values()),
        "checks": checks,
        "models": settings.list_transcription_models(),
        "defaults": {
            "language": settings.default_language,
            "beam_size": settings.beam_size,
            "best_of": settings.best_of,
            "chunk_seconds": settings.chunk_seconds,
            "chunk_overlap_seconds": settings.chunk_overlap_seconds,
            "stitch_method": settings.stitch_method,
            "repetition_guard": settings.repetition_guard,
        },
    }


@app.get("/models")
def models() -> dict:
    return {"models": settings.list_transcription_models()}


@app.get("/stitch-methods")
def stitch_methods() -> dict:
    return {
        "methods": sorted(stitch_utils.STITCH_METHODS),
        "default": settings.stitch_method,
    }


@app.post("/jobs/transcribe/upload", status_code=202)
async def transcribe_upload(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()],
    language: Annotated[str | None, Form()] = None,
    beam_size: Annotated[int | None, Form(ge=1)] = None,
    best_of: Annotated[int | None, Form(ge=1)] = None,
    vad_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.1,
    vad_max_speech_duration_s: Annotated[int, Form(ge=1)] = 3600,
    vad_min_silence_duration_ms: Annotated[int, Form(ge=0)] = 2000,
    vad_speech_pad_ms: Annotated[int, Form(ge=0)] = 400,
    chunk_seconds: Annotated[int | None, Form(ge=0)] = None,
    chunk_overlap_seconds: Annotated[int | None, Form(ge=0)] = None,
    stitch_method: Annotated[str | None, Form(pattern=STITCH_METHOD_PATTERN)] = None,
    repetition_guard: Annotated[bool | None, Form()] = None,
) -> dict:
    resolved_chunk_seconds = settings.chunk_seconds if chunk_seconds is None else chunk_seconds
    metadata = await job_store.create_upload_job(
        upload=file,
        model=model,
        language=language or settings.default_language,
        beam_size=beam_size or settings.beam_size,
        best_of=best_of or settings.best_of,
        vad_threshold=vad_threshold,
        vad_max_speech_duration_s=vad_max_speech_duration_s,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        chunk_seconds=resolved_chunk_seconds,
        chunk_overlap_seconds=(
            settings.chunk_overlap_seconds if chunk_overlap_seconds is None else chunk_overlap_seconds
        ),
        stitch_method=stitch_method if stitch_method is not None else (
            settings.stitch_method if resolved_chunk_seconds > 0 else None
        ),
        repetition_guard=settings.repetition_guard if repetition_guard is None else repetition_guard,
    )
    return metadata


@app.post("/jobs/transcribe/path", status_code=202)
def transcribe_path(request: PathTranscriptionRequest) -> dict:
    resolved_chunk_seconds = settings.chunk_seconds if request.chunk_seconds is None else request.chunk_seconds
    return job_store.create_path_job(
        path=Path(request.path),
        model=request.model,
        language=request.language or settings.default_language,
        beam_size=request.beam_size or settings.beam_size,
        best_of=request.best_of or settings.best_of,
        vad_threshold=request.vad_threshold,
        vad_max_speech_duration_s=request.vad_max_speech_duration_s,
        vad_min_silence_duration_ms=request.vad_min_silence_duration_ms,
        vad_speech_pad_ms=request.vad_speech_pad_ms,
        chunk_seconds=resolved_chunk_seconds,
        chunk_overlap_seconds=(
            settings.chunk_overlap_seconds
            if request.chunk_overlap_seconds is None
            else request.chunk_overlap_seconds
        ),
        stitch_method=request.stitch_method if request.stitch_method is not None else (
            settings.stitch_method if resolved_chunk_seconds > 0 else None
        ),
        repetition_guard=settings.repetition_guard if request.repetition_guard is None else request.repetition_guard,
        stitch_methods=request.stitch_methods,
    )


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return job_store.list_terminal_jobs()


@app.delete("/jobs", status_code=204)
def clear_jobs() -> Response:
    job_store.clear_jobs()
    return Response(status_code=204)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return job_store.get_job(job_id)


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str) -> dict:
    return job_store.get_result(job_id)


@app.get("/jobs/{job_id}/transcript.md")
def get_transcript_markdown(job_id: str) -> Response:
    markdown = job_store.get_transcript_markdown(job_id)
    filename = job_store.transcript_download_name(job_id)
    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/jobs/{job_id}/llm-prompt.txt")
def get_llm_prompt(job_id: str) -> Response:
    prompt = job_store.get_llm_prompt(job_id)
    filename = job_store.llm_prompt_download_name(job_id)
    return Response(
        content=prompt,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    job_store.delete_job(job_id)
    return Response(status_code=204)


def _path_check(path: Path, executable: bool = False) -> dict:
    ok = path.exists() and path.is_file()
    if executable:
        ok = ok and path.stat().st_mode & 0o111 != 0
    return {"path": str(path), "ok": bool(ok)}


def _binary_check(name: str) -> dict:
    path = shutil.which(name)
    return {"path": path, "ok": path is not None}
