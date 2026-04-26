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


@dataclass(frozen=True)
class Settings:
    whispercpp_bin: Path
    whispercpp_model: Path
    whispercpp_vad_model: Path
    temp_dir: Path
    default_language: str
    beam_size: int
    best_of: int

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
            os.getenv("WHISPERCPP_MODEL", "/home/transcribe/whisper.cpp/models/ggml-large-v3-q5_0.bin")
        ),
        whispercpp_vad_model=Path(
            os.getenv("WHISPERCPP_VAD_MODEL", "/home/transcribe/whisper.cpp/models/ggml-silero-v6.2.0.bin")
        ),
        temp_dir=Path(os.getenv("WHISPERCPP_TEMP_DIR", str(default_temp_dir))),
        default_language=os.getenv("WHISPERCPP_DEFAULT_LANGUAGE", "it"),
        beam_size=_env_int("WHISPERCPP_BEAM_SIZE", 3),
        best_of=_env_int("WHISPERCPP_BEST_OF", 3),
    )
