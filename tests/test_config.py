import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app.config import Settings, get_settings


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "whisper.cpp"
        self.bin = self.base / "build" / "bin" / "whisper-cli"
        self.models = self.base / "models"
        self.bin.parent.mkdir(parents=True)
        self.models.mkdir(parents=True)
        self.bin.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        self.bin.chmod(0o755)
        (self.models / "ggml-large-v3.bin").write_bytes(b"model")
        (self.models / "ggml-large-v3-turbo.bin").write_bytes(b"model")
        (self.models / "ggml-silero-v6.2.0.bin").write_bytes(b"vad")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_get_settings_derives_whispercpp_paths_from_base_dir(self) -> None:
        with patch.dict(os.environ, {"WHISPERCPP_BASE_DIR": str(self.base)}, clear=False):
            settings = get_settings()
            settings.validate_startup()

        self.assertEqual(settings.whispercpp_base_dir, self.base)
        self.assertEqual(settings.whispercpp_bin, self.bin)
        self.assertEqual(settings.whispercpp_models_dir, self.models)
        self.assertEqual(settings.whispercpp_vad_model, self.models / "ggml-silero-v6.2.0.bin")

    def test_list_transcription_models_excludes_vad_models(self) -> None:
        settings = self._settings()

        self.assertEqual(
            settings.list_transcription_models(),
            ["ggml-large-v3-turbo.bin", "ggml-large-v3.bin"],
        )

    def test_get_settings_validates_stitch_method_from_shared_choices(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WHISPERCPP_BASE_DIR": str(self.base),
                "WHISPERCPP_STITCH_METHOD": "safe_zone",
            },
            clear=False,
        ):
            settings = get_settings()

        self.assertEqual(settings.stitch_method, "safe_zone")

        with patch.dict(
            os.environ,
            {
                "WHISPERCPP_BASE_DIR": str(self.base),
                "WHISPERCPP_STITCH_METHOD": "missing",
            },
            clear=False,
        ):
            settings = get_settings()

        self.assertEqual(settings.stitch_method, "fuzzy")

    def test_resolve_model_rejects_unknown_or_unsafe_names(self) -> None:
        settings = self._settings()

        self.assertEqual(settings.resolve_model("ggml-large-v3.bin"), self.models / "ggml-large-v3.bin")
        for model in ["missing.bin", "../ggml-large-v3.bin", str(self.models / "ggml-large-v3.bin")]:
            with self.assertRaises(HTTPException):
                settings.resolve_model(model)

    def test_validate_startup_requires_compiled_binary_and_models(self) -> None:
        self.bin.unlink()
        settings = self._settings()

        with self.assertRaisesRegex(RuntimeError, "whispercpp_bin"):
            settings.validate_startup()

    def _settings(self) -> Settings:
        return Settings(
            whispercpp_base_dir=self.base,
            whispercpp_bin=self.bin,
            whispercpp_models_dir=self.models,
            whispercpp_vad_model=self.models / "ggml-silero-v6.2.0.bin",
            temp_dir=self.root,
            default_language="it",
            beam_size=3,
            best_of=3,
            chunking_mode="auto",
            chunk_threshold_seconds=1800,
            chunk_seconds=1800,
            chunk_overlap_seconds=30,
            stitch_method="fuzzy",
            repetition_guard=True,
        )
