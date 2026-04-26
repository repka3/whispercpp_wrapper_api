import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    whispercpp_bin: Path
    whispercpp_model: Path
    whispercpp_vad_model: Path
    temp_dir: Path
    default_language: str
    beam_size: int
    best_of: int
    chunking_mode: str
    chunk_threshold_seconds: int
    chunk_seconds: int
    chunk_overlap_seconds: int
    repetition_guard: bool

    @property
    def jobs_dir(self) -> Path:
        return self.temp_dir / "jobs"


def get_settings() -> Settings:
    default_temp_dir = PROJECT_ROOT
    return Settings(
        whispercpp_bin=Path(
            os.getenv("WHISPERCPP_BIN", "/home/transcribe/whisper.cpp/build/bin/whisper-cli")
        ),
        whispercpp_model=Path(
            os.getenv("WHISPERCPP_MODEL", "/home/transcribe/whisper.cpp/models/ggml-large-v3-q8_0.bin")
        ),
        whispercpp_vad_model=Path(
            os.getenv("WHISPERCPP_VAD_MODEL", "/home/transcribe/whisper.cpp/models/ggml-silero-v6.2.0.bin")
        ),
        temp_dir=Path(os.getenv("WHISPERCPP_TEMP_DIR", str(default_temp_dir))),
        default_language=os.getenv("WHISPERCPP_DEFAULT_LANGUAGE", "it"),
        beam_size=_env_int("WHISPERCPP_BEAM_SIZE", 3),
        best_of=_env_int("WHISPERCPP_BEST_OF", 3),
        chunking_mode=_env_choice("WHISPERCPP_CHUNKING_MODE", "auto", {"off", "auto", "always"}),
        chunk_threshold_seconds=max(_env_int("WHISPERCPP_CHUNK_THRESHOLD_SECONDS", 1800), 1),
        chunk_seconds=max(_env_int("WHISPERCPP_CHUNK_SECONDS", 1800), 1),
        chunk_overlap_seconds=max(_env_int("WHISPERCPP_CHUNK_OVERLAP_SECONDS", 30), 0),
        repetition_guard=_env_bool("WHISPERCPP_REPETITION_GUARD", True),
    )
