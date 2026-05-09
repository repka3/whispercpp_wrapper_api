import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

import app.main as main
from app.config import Settings
from app.jobs import JobStore


class ApiModelTests(unittest.TestCase):
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
        (self.models / "ggml-silero-v6.2.0.bin").write_bytes(b"vad")
        self.audio = self.root / "audio.wav"
        self.audio.write_bytes(b"audio")
        self.settings = Settings(
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

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_models_endpoint_lists_transcription_models(self) -> None:
        with self._patch_app():
            response = main.models()

        self.assertEqual(response, {"models": ["ggml-large-v3.bin"]})

    def test_path_transcription_requires_model(self) -> None:
        with self.assertRaises(ValidationError):
            main.PathTranscriptionRequest(path=str(self.audio))

    def test_path_transcription_defaults_vad_threshold_to_point_one(self) -> None:
        request = main.PathTranscriptionRequest(path=str(self.audio), model="ggml-large-v3.bin")

        self.assertEqual(request.vad_threshold, 0.1)

    def test_path_transcription_accepts_stitch_method(self) -> None:
        request = main.PathTranscriptionRequest(
            path=str(self.audio),
            model="ggml-large-v3.bin",
            stitch_method="safe_zone",
        )

        self.assertEqual(request.stitch_method, "safe_zone")

    def test_path_transcription_rejects_unknown_model(self) -> None:
        with self._patch_app():
            with self.assertRaises(HTTPException) as context:
                main.transcribe_path(
                    main.PathTranscriptionRequest(
                        path=str(self.audio),
                        model="missing.bin",
                    )
                )

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("Unknown transcription model", context.exception.detail)

    def test_path_transcription_rejects_traversal_model(self) -> None:
        with self._patch_app():
            with self.assertRaises(HTTPException) as context:
                main.transcribe_path(
                    main.PathTranscriptionRequest(
                        path=str(self.audio),
                        model="../ggml-large-v3.bin",
                    )
                )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.detail, "Model must be a filename from /models")

    def _patch_app(self):
        return patch.multiple(main, settings=self.settings, job_store=JobStore(self.settings))
