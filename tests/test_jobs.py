import json
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from app.config import Settings
from app.jobs import JobStore


class JobStoreManagementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = JobStore(
            Settings(
                whispercpp_base_dir=root,
                whispercpp_bin=root / "whisper-cli",
                whispercpp_models_dir=root,
                whispercpp_vad_model=root / "vad.bin",
                temp_dir=root,
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
        )
        self.store.initialize()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_list_terminal_jobs_sorts_and_includes_failed(self) -> None:
        self._write_job(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            status="succeeded",
            completed_at="2026-04-26T10:00:00+00:00",
            filename="old.mp3",
            model="model-from-metadata.bin",
            metrics={"rtf": 0.5},
        )
        self._write_job(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            status="failed",
            completed_at="2026-04-26T11:00:00+00:00",
            filename="broken.mp3",
            error="whisper failed",
        )
        self._write_job(
            "cccccccccccccccccccccccccccccccc",
            status="running",
            completed_at=None,
            filename="running.mp3",
        )

        jobs = self.store.list_terminal_jobs()

        self.assertEqual([item["job_id"] for item in jobs], [
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ])
        self.assertEqual(jobs[0]["source"]["filename"], "broken.mp3")
        self.assertEqual(jobs[0]["error"], "whisper failed")
        self.assertEqual(jobs[1]["model"], "model-from-metadata.bin")
        self.assertEqual(jobs[1]["metrics"], {"rtf": 0.5})

    def test_list_terminal_jobs_falls_back_to_result_model(self) -> None:
        job_dir = self._write_job(
            "dddddddddddddddddddddddddddddddd",
            status="succeeded",
            completed_at="2026-04-26T12:00:00+00:00",
            filename="legacy.mp3",
            model=None,
        )
        self._write_result(job_dir, model="model-from-result.bin")

        jobs = self.store.list_terminal_jobs()

        self.assertEqual(jobs[0]["model"], "model-from-result.bin")

    def test_transcript_markdown_requires_success_and_renders_segments(self) -> None:
        job_dir = self._write_job(
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            status="succeeded",
            completed_at="2026-04-26T12:30:00+00:00",
            filename="meeting.mp3",
            model="model.bin",
            metrics={"audio_duration_seconds": 2.0, "elapsed_seconds": 1.0, "rtf": 0.5, "speedup": 2.0},
        )
        self._write_result(job_dir, model="model.bin")
        self._write_job(
            "ffffffffffffffffffffffffffffffff",
            status="failed",
            completed_at="2026-04-26T12:31:00+00:00",
            filename="failed.mp3",
            error="decode failed",
        )

        markdown = self.store.get_transcript_markdown("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")

        self.assertIn("# Transcript: meeting.mp3", markdown)
        self.assertIn("- Model: model.bin", markdown)
        self.assertIn("\n## Transcript\n\nCiao mondo.\n", markdown)
        self.assertNotIn("00:00:00.000", markdown)
        with self.assertRaises(HTTPException) as context:
            self.store.get_transcript_markdown("ffffffffffffffffffffffffffffffff")
        self.assertEqual(context.exception.status_code, 409)

    def test_llm_prompt_is_compact_and_requires_success(self) -> None:
        job_dir = self._write_job(
            "11111111111111111111111111111111",
            status="succeeded",
            completed_at="2026-04-26T12:30:00+00:00",
            filename="meeting.mp3",
            model="model.bin",
        )
        self._write_result(job_dir, model="model.bin", text="Ciao   mondo.\n\nSeconda    frase.")
        self._write_job(
            "22222222222222222222222222222222",
            status="failed",
            completed_at="2026-04-26T12:31:00+00:00",
            filename="failed.mp3",
            error="decode failed",
        )

        prompt = self.store.get_llm_prompt("11111111111111111111111111111111")

        self.assertIn("Summarize the following meeting transcript.", prompt)
        self.assertIn("Transcript:\nCiao mondo. Seconda frase.\n", prompt)
        self.assertNotIn("model.bin", prompt)
        self.assertNotIn("00:00:00.000", prompt)
        with self.assertRaises(HTTPException) as context:
            self.store.get_llm_prompt("22222222222222222222222222222222")
        self.assertEqual(context.exception.status_code, 409)

    def test_delete_job_removes_directory(self) -> None:
        job_dir = self._write_job(
            "99999999999999999999999999999999",
            status="failed",
            completed_at="2026-04-26T13:00:00+00:00",
            filename="delete-me.mp3",
        )

        self.store.delete_job("99999999999999999999999999999999")

        self.assertFalse(job_dir.exists())

    def _write_job(
        self,
        job_id: str,
        *,
        status: str,
        completed_at: str | None,
        filename: str,
        model: str | None = None,
        metrics: dict | None = None,
        error: str | None = None,
    ) -> Path:
        job_dir = self.store.settings.jobs_dir / job_id
        job_dir.mkdir(parents=True)
        result_path = job_dir / "result.json"
        metadata = {
            "job_id": job_id,
            "status": status,
            "progress": 100 if status == "succeeded" else 99,
            "created_at": "2026-04-26T09:00:00+00:00",
            "updated_at": completed_at or "2026-04-26T09:00:00+00:00",
            "started_at": "2026-04-26T09:01:00+00:00",
            "completed_at": completed_at,
            "source": {
                "kind": "upload",
                "path": str(job_dir / filename),
                "filename": filename,
                "size_bytes": 1234,
            },
            "params": {
                "language": "it",
                "beam_size": 3,
                "best_of": 3,
                "chunking": {
                    "mode": "auto",
                    "chunk_seconds": 1800,
                    "overlap_seconds": 30,
                    "repetition_guard": True,
                },
            },
            "logs": {},
            "result_path": str(result_path) if status == "succeeded" else None,
            "error": error,
            "metrics": metrics or {},
        }
        if model is not None:
            metadata["model"] = model
        self.store._write_metadata(job_dir, metadata)
        return job_dir

    def _write_result(self, job_dir: Path, *, model: str, text: str = "Ciao mondo.") -> None:
        result = {
            "job_id": job_dir.name,
            "engine": "whisper.cpp",
            "model": model,
            "language": "it",
            "text": text,
            "segments": [
                {
                    "start": 0,
                    "end": 1.25,
                    "transcript": "Ciao mondo.",
                    "words": [],
                }
            ],
            "decode": {"beam_size": 3, "best_of": 3},
            "metrics": {
                "audio_duration_seconds": 2.0,
                "elapsed_seconds": 1.0,
                "rtf": 0.5,
                "speedup": 2.0,
            },
        }
        with (job_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle)


if __name__ == "__main__":
    unittest.main()
