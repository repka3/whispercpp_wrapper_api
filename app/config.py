import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    value = raw.lower()
    if value in choices:
        return value
    return default


@dataclass(frozen=True)
class Settings:
    whispercpp_base_dir: Path
    whispercpp_bin: Path
    whispercpp_models_dir: Path
    whispercpp_vad_model: Path
    temp_dir: Path
    default_language: str
    beam_size: int
    best_of: int
    chunking_mode: str
    chunk_threshold_seconds: int
    chunk_seconds: int
    chunk_overlap_seconds: int
    stitch_method: str
    repetition_guard: bool

    @property
    def jobs_dir(self) -> Path:
        return self.temp_dir / "jobs"

    def list_transcription_models(self) -> list[str]:
        if not self.whispercpp_models_dir.exists() or not self.whispercpp_models_dir.is_dir():
            return []
        models = []
        for path in self.whispercpp_models_dir.glob("ggml*.bin"):
            name = path.name
            lowered = name.lower()
            if "silero" in lowered or "vad" in lowered:
                continue
            if path.is_file():
                models.append(name)
        return sorted(models)

    def resolve_model(self, model: str) -> Path:
        if not model or Path(model).name != model or Path(model).is_absolute():
            raise HTTPException(status_code=400, detail="Model must be a filename from /models")
        path = self.whispercpp_models_dir / model
        if model not in self.list_transcription_models() or not path.exists() or not path.is_file():
            raise HTTPException(status_code=400, detail=f"Unknown transcription model: {model}")
        return path

    def startup_checks(self) -> dict[str, dict]:
        models = self.list_transcription_models()
        return {
            "whispercpp_base_dir": {
                "path": str(self.whispercpp_base_dir),
                "ok": self.whispercpp_base_dir.exists() and self.whispercpp_base_dir.is_dir(),
            },
            "whispercpp_bin": {
                "path": str(self.whispercpp_bin),
                "ok": self.whispercpp_bin.exists()
                and self.whispercpp_bin.is_file()
                and self.whispercpp_bin.stat().st_mode & 0o111 != 0,
            },
            "models_dir": {
                "path": str(self.whispercpp_models_dir),
                "ok": self.whispercpp_models_dir.exists() and self.whispercpp_models_dir.is_dir() and bool(models),
            },
            "vad_model": {
                "path": str(self.whispercpp_vad_model),
                "ok": self.whispercpp_vad_model.exists() and self.whispercpp_vad_model.is_file(),
            },
        }

    def validate_startup(self) -> None:
        missing_base = not os.getenv("WHISPERCPP_BASE_DIR")
        checks = self.startup_checks()
        failures = [name for name, check in checks.items() if not check["ok"]]
        if missing_base:
            failures.insert(0, "WHISPERCPP_BASE_DIR")
        if failures:
            details = ", ".join(dict.fromkeys(failures))
            raise RuntimeError(
                f"Invalid whisper.cpp configuration: {details}. "
                "Set WHISPERCPP_BASE_DIR in .env and ensure whisper-cli is compiled and models are downloaded."
            )


def get_settings() -> Settings:
    default_temp_dir = PROJECT_ROOT
    whispercpp_base_dir = Path(os.getenv("WHISPERCPP_BASE_DIR", ""))
    whispercpp_models_dir = whispercpp_base_dir / "models"
    vad_model = _find_vad_model(whispercpp_models_dir)
    return Settings(
        whispercpp_base_dir=whispercpp_base_dir,
        whispercpp_bin=whispercpp_base_dir / "build" / "bin" / "whisper-cli",
        whispercpp_models_dir=whispercpp_models_dir,
        whispercpp_vad_model=vad_model or whispercpp_models_dir / "ggml-silero-v6.2.0.bin",
        temp_dir=Path(os.getenv("WHISPERCPP_TEMP_DIR", str(default_temp_dir))),
        default_language=os.getenv("WHISPERCPP_DEFAULT_LANGUAGE", "it"),
        beam_size=_env_int("WHISPERCPP_BEAM_SIZE", 3),
        best_of=_env_int("WHISPERCPP_BEST_OF", 3),
        chunking_mode=_env_choice("WHISPERCPP_CHUNKING_MODE", "auto", {"off", "auto", "always"}),
        chunk_threshold_seconds=max(_env_int("WHISPERCPP_CHUNK_THRESHOLD_SECONDS", 1800), 1),
        chunk_seconds=max(_env_int("WHISPERCPP_CHUNK_SECONDS", 1800), 1),
        chunk_overlap_seconds=max(_env_int("WHISPERCPP_CHUNK_OVERLAP_SECONDS", 30), 0),
        stitch_method=_env_choice("WHISPERCPP_STITCH_METHOD", "fuzzy", {"fuzzy", "safe_zone"}),
        repetition_guard=_env_bool("WHISPERCPP_REPETITION_GUARD", True),
    )


def _find_vad_model(models_dir: Path) -> Path | None:
    if not models_dir.exists() or not models_dir.is_dir():
        return None
    for pattern in ("ggml-silero*.bin", "*silero*.bin", "*vad*.bin"):
        matches = sorted(path for path in models_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None
