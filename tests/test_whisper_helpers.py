import unittest
from pathlib import Path
from unittest.mock import patch

from app.whisper import (
    build_stitch_variants,
    build_chunk_windows,
    build_chunk_window_plan,
    build_mixed_aligned_chunk_windows,
    build_silence_aligned_chunk_windows,
    detect_repetition,
    parse_vad_speech_segments,
    run_transcription,
    VadSpeechSegment,
    _extract_segments,
)
from app.config import Settings
from app.stitch_utils import (
    format_timestamp,
    merge_chunk_segments,
    merge_chunk_segments_with_audit,
    render_stitch_audit_markdown,
)


class WhisperHelperTests(unittest.TestCase):
    def _segment(self, start: float, end: float, transcript: str) -> dict:
        return {"start": start, "end": end, "transcript": transcript, "words": []}

    def _word_segment(self, start: float, end: float, transcript: str) -> dict:
        words = []
        for index, word in enumerate(transcript.split()):
            words.append(
                {
                    "word": word,
                    "start": round(start + index, 3),
                    "end": round(start + index + 0.5, 3),
                }
            )
        return {"start": start, "end": end, "transcript": transcript, "words": words}

    def test_build_chunk_windows_with_partial_final_chunk(self) -> None:
        windows = build_chunk_windows(
            audio_duration_seconds=3700,
            chunk_seconds=1800,
            overlap_seconds=30,
        )

        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].start_seconds, 0)
        self.assertEqual(windows[1].start_seconds, 1770)
        self.assertEqual(windows[2].start_seconds, 3540)
        self.assertEqual(windows[2].duration_seconds, 160)

    def test_build_chunk_windows_clamps_large_overlap(self) -> None:
        windows = build_chunk_windows(
            audio_duration_seconds=3,
            chunk_seconds=2,
            overlap_seconds=5,
        )

        self.assertEqual([item.start_seconds for item in windows], [0, 1])

    def test_parse_vad_speech_segments_reads_helper_output(self) -> None:
        segments = parse_vad_speech_segments(
            "\n".join(
                [
                    "Detected 2 speech segments:",
                    "Speech segment 0: start = 29.00, end = 221.00",
                    "Speech segment 1: start = 330.00, end = 377.00",
                ]
            )
        )

        self.assertEqual(
            segments,
            [
                VadSpeechSegment(start_seconds=0.29, end_seconds=2.21),
                VadSpeechSegment(start_seconds=3.3, end_seconds=3.77),
            ],
        )

    def test_parse_vad_speech_segments_prefers_second_based_vad_logs(self) -> None:
        segments = parse_vad_speech_segments(
            "\n".join(
                [
                    "Detected 1 speech segments:",
                    "Speech segment 0: start = 0.00, end = 2165.00",
                    "whisper_vad_segments_from_probs: VAD segment 0: start = 0.00, end = 21.65 (duration: 21.65)",
                ]
            )
        )

        self.assertEqual(segments, [VadSpeechSegment(start_seconds=0, end_seconds=21.65)])

    def test_silence_aligned_chunk_windows_use_absolute_targets_without_drift(self) -> None:
        windows, decisions = build_silence_aligned_chunk_windows(
            audio_duration_seconds=750,
            chunk_seconds=300,
            speech_segments=[
                VadSpeechSegment(start_seconds=0, end_seconds=390),
                VadSpeechSegment(start_seconds=410, end_seconds=590),
                VadSpeechSegment(start_seconds=610, end_seconds=750),
            ],
            silence_min_duration_ms=1000,
        )

        self.assertEqual([item.selected_seconds for item in decisions], [400, 600])
        self.assertEqual([item.target_seconds for item in decisions], [300, 600])
        self.assertEqual([item.start_seconds for item in windows], [0, 400, 600])
        self.assertEqual([item.duration_seconds for item in windows], [400, 200, 150])

    def test_mixed_chunk_windows_fall_back_per_boundary(self) -> None:
        windows, decisions = build_mixed_aligned_chunk_windows(
            audio_duration_seconds=750,
            chunk_seconds=300,
            speech_segments=[
                VadSpeechSegment(start_seconds=0, end_seconds=450),
                VadSpeechSegment(start_seconds=550, end_seconds=590),
                VadSpeechSegment(start_seconds=610, end_seconds=750),
            ],
            silence_min_duration_ms=1000,
            overlap_seconds=30,
        )

        self.assertEqual([item.cut_type for item in decisions], ["hard_fallback", "vad_silence"])
        self.assertEqual([item.selected_seconds for item in decisions], [300, 600])
        self.assertEqual([item.start_seconds for item in windows], [0, 270, 600])
        self.assertEqual([item.previous_overlap_seconds for item in windows], [0, 30, 0])
        self.assertEqual([item.next_overlap_seconds for item in windows], [30, 0, 0])

    def test_chunk_window_plan_uses_vad_silence_with_zero_overlap(self) -> None:
        import tempfile

        vad_output = "\n".join(
            [
                "Detected 3 speech segments:",
                "Speech segment 0: start = 0.00, end = 39000.00",
                "Speech segment 1: start = 41000.00, end = 59000.00",
                "Speech segment 2: start = 61000.00, end = 75000.00",
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.whisper.run_vad_speech_segments",
            return_value=vad_output,
        ):
            root = Path(temp_dir)
            plan = build_chunk_window_plan(
                input_path=root / "audio.wav",
                settings=Settings(
                    whispercpp_base_dir=root,
                    whispercpp_bin=root / "whisper-cli",
                    whispercpp_models_dir=root,
                    whispercpp_vad_model=root / "vad.bin",
                    temp_dir=root,
                    default_language="it",
                    beam_size=3,
                    best_of=3,
                    chunk_seconds=300,
                    chunk_overlap_seconds=30,
                    stitch_method="fuzzy",
                    repetition_guard=True,
                ),
                audio_duration_seconds=750,
                chunk_seconds=300,
                chunk_overlap_seconds=30,
                vad_threshold=0.1,
                vad_max_speech_duration_s=3600,
                vad_min_silence_duration_ms=1000,
                vad_speech_pad_ms=400,
            )

        self.assertEqual(plan.strategy, "vad_silence")
        self.assertEqual(plan.overlap_seconds, 0)
        self.assertEqual([item.start_seconds for item in plan.windows], [0, 400, 600])
        self.assertEqual(plan.warnings, [])

    def test_chunk_window_plan_falls_back_to_fixed_windows_when_vad_fails(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.whisper.run_vad_speech_segments",
            side_effect=OSError("helper missing"),
        ):
            root = Path(temp_dir)
            plan = build_chunk_window_plan(
                input_path=root / "audio.wav",
                settings=Settings(
                    whispercpp_base_dir=root,
                    whispercpp_bin=root / "whisper-cli",
                    whispercpp_models_dir=root,
                    whispercpp_vad_model=root / "vad.bin",
                    temp_dir=root,
                    default_language="it",
                    beam_size=3,
                    best_of=3,
                    chunk_seconds=1800,
                    chunk_overlap_seconds=30,
                    stitch_method="fuzzy",
                    repetition_guard=True,
                ),
                audio_duration_seconds=3700,
                chunk_seconds=1800,
                chunk_overlap_seconds=30,
                vad_threshold=0.1,
                vad_max_speech_duration_s=3600,
                vad_min_silence_duration_ms=2000,
                vad_speech_pad_ms=400,
            )

        self.assertEqual(plan.strategy, "fixed_fallback")
        self.assertEqual(plan.overlap_seconds, 30)
        self.assertEqual([item.start_seconds for item in plan.windows], [0, 1770, 3540])
        self.assertEqual(plan.warnings[0]["type"], "chunking_fixed_fallback")

    def test_build_stitch_variants_reuses_chunk_segments_for_each_method(self) -> None:
        windows = build_chunk_windows(
            audio_duration_seconds=4,
            chunk_seconds=3,
            overlap_seconds=1,
        )
        with self.subTest("fixture"):
            self.assertEqual(len(windows), 2)

        chunk_outputs = [
            {
                "window": windows[0],
                "segments": [self._segment(0, 2, "Ciao mondo")],
                "warning": None,
            },
            {
                "window": windows[1],
                "segments": [
                    self._segment(2, 2.5, "Ciao mondo"),
                    self._segment(3, 4, "Nuovo testo"),
                ],
                "warning": None,
            },
        ]

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            variants = build_stitch_variants(
                job_id="job-1",
                job_dir=Path(temp_dir),
                chunk_outputs=chunk_outputs,
                stitch_methods=["fuzzy", "safe_zone"],
                chunk_overlap_seconds=1,
                chunk_seconds=3,
                repetition_guard=True,
                retried_chunks=[],
                warnings=[],
                log_stitch=None,
                log_primary_method="fuzzy",
            )

        self.assertEqual(set(variants), {"fuzzy", "safe_zone"})
        self.assertIn("Nuovo testo", variants["fuzzy"]["text"])
        self.assertEqual(variants["fuzzy"]["decode"]["chunking"]["stitch_method"], "fuzzy")
        self.assertEqual(variants["safe_zone"]["decode"]["chunking"]["stitch_method"], "safe_zone")

    def test_single_pass_transcription_does_not_create_stitch_variants(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "audio.wav"
            input_path.write_bytes(b"audio")
            model_path = root / "model.bin"
            model_path.write_bytes(b"model")
            settings = Settings(
                whispercpp_base_dir=root,
                whispercpp_bin=root / "whisper-cli",
                whispercpp_models_dir=root,
                whispercpp_vad_model=root / "vad.bin",
                temp_dir=root,
                default_language="it",
                beam_size=3,
                best_of=3,
                chunk_seconds=0,
                chunk_overlap_seconds=30,
                stitch_method="fuzzy",
                repetition_guard=True,
            )
            (root / "whisper_output.json").write_text("{}", encoding="utf-8")
            base_result = {
                "job_id": "job-1",
                "engine": "whisper.cpp",
                "model": "model.bin",
                "language": "it",
                "text": "Ciao mondo",
                "segments": [],
                "decode": {"beam_size": 3, "best_of": 3},
                "metrics": {},
            }

            with patch("app.whisper.probe_duration_seconds", return_value=10.0), \
                patch("app.whisper._start_whisper", return_value=object()), \
                patch("app.whisper._capture_process", return_value=0), \
                patch("app.whisper.normalize_whisper_json", return_value=base_result):
                result = run_transcription(
                    job_id="job-1",
                    job_dir=root,
                    input_path=input_path,
                    settings=settings,
                    model_path=model_path,
                    language="it",
                    beam_size=3,
                    best_of=3,
                    vad_threshold=0.1,
                    vad_max_speech_duration_s=3600,
                    vad_min_silence_duration_ms=2000,
                    vad_speech_pad_ms=400,
                    chunk_seconds=0,
                    chunk_overlap_seconds=30,
                    stitch_method="safe_zone",
                    stitch_methods=["fuzzy", "safe_zone"],
                    repetition_guard=True,
                    set_progress=lambda _progress: None,
                )

        self.assertNotIn("stitch_variants", result)
        self.assertEqual(
            result["decode"]["chunking"],
            {
                "enabled": False,
                "chunk_seconds": 0,
                "overlap_seconds": 0,
                "stitch_method": None,
                "stitch_methods": None,
            },
        )

    def test_merge_chunk_segments_dedupes_overlap(self) -> None:
        existing = [
            self._segment(0, 10, "Ciao mondo."),
        ]
        incoming = [
            self._segment(9, 11, "ciao mondo"),
            self._segment(12, 13, "Nuovo testo"),
        ]

        merged = merge_chunk_segments(
            existing,
            incoming,
            overlap_start_seconds=9,
            overlap_end_seconds=11,
        )

        self.assertEqual([item["transcript"] for item in merged], ["Ciao mondo.", "Nuovo testo"])

    def test_merge_with_audit_records_dropped_duplicate(self) -> None:
        existing = [
            self._segment(0, 10, "Ciao mondo."),
        ]
        incoming = [
            self._segment(9, 11, "ciao mondo"),
            self._segment(12, 13, "Nuovo testo"),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            existing,
            incoming,
            previous_chunk_index=0,
            next_chunk_index=1,
            overlap_start_seconds=9,
            overlap_end_seconds=11,
            incoming_warning=None,
        )

        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual([item["transcript"] for item in merged], ["Ciao mondo.", "Nuovo testo"])
        self.assertEqual(audit["counts"]["dropped_duplicates"], 1)
        self.assertEqual(audit["counts"]["dropped_exact_duplicates"], 1)
        self.assertEqual(audit["incoming_head"][0]["decision"], "dropped_exact_duplicate")

    def test_merge_with_audit_keeps_disagreement_inside_overlap(self) -> None:
        existing = [
            self._segment(0, 10, "Primo testo affidabile"),
        ]
        incoming = [
            self._segment(9, 11, "Testo diverso davvero"),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            existing,
            incoming,
            previous_chunk_index=0,
            next_chunk_index=1,
            overlap_start_seconds=9,
            overlap_end_seconds=11,
            incoming_warning=None,
        )

        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual([item["transcript"] for item in merged], ["Primo testo affidabile", "Testo diverso davvero"])
        self.assertEqual(audit["incoming_head"][0]["decision"], "kept_low_confidence_overlap")
        self.assertEqual(audit["counts"]["kept_overlap"], 1)
        self.assertEqual(audit["counts"]["kept_low_confidence_overlap"], 1)

    def test_merge_drops_split_duplicate_segments(self) -> None:
        existing = [
            self._segment(
                0,
                12,
                (
                    "Una campagna informativa tempestiva puo evitare congestioni presso gli sportelli, "
                    "disagio alla cittadinanza, ritardi nel rilascio dei documenti."
                ),
            ),
        ]
        incoming = [
            self._segment(9, 10, "Una campagna informativa tempestiva puo evitare congestioni presso gli sportelli,"),
            self._segment(10, 11, "disagio alla cittadinanza, ritardi nel rilascio dei documenti."),
            self._segment(12, 13, "Nuova frase."),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            existing,
            incoming,
            previous_chunk_index=0,
            next_chunk_index=1,
            overlap_start_seconds=9,
            overlap_end_seconds=12,
            incoming_warning=None,
        )

        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual([item["transcript"] for item in merged], [existing[0]["transcript"], "Nuova frase."])
        self.assertEqual(audit["counts"]["dropped_overlap_duplicates"], 2)
        self.assertEqual(audit["incoming_head"][0]["decision"], "dropped_overlap_duplicate")

    def test_merge_drops_merged_duplicate_segment(self) -> None:
        existing = [
            self._segment(0, 6, "La relativa gara e stata vinta da FiberCop."),
            self._segment(6, 10, "Sta portando avanti i diversi cantieri."),
        ]
        incoming = [
            self._segment(
                9,
                11,
                "La relativa gara e stata vinta da FiberCop. Sta portando avanti i diversi cantieri.",
            ),
            self._segment(11, 12, "Nuovo testo."),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            existing,
            incoming,
            previous_chunk_index=0,
            next_chunk_index=1,
            overlap_start_seconds=9,
            overlap_end_seconds=11,
            incoming_warning=None,
        )

        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual([item["transcript"] for item in merged], [item["transcript"] for item in existing] + ["Nuovo testo."])
        self.assertEqual(audit["counts"]["dropped_overlap_duplicates"], 1)

    def test_merge_trims_boundary_crossing_duplicate_prefix(self) -> None:
        existing = [
            self._segment(0, 10, "Io credo, guardi, non ho paura a dire"),
        ]
        incoming = [
            self._segment(9, 20, "Io credo non ho paura a dire nomi e cognomi e zone"),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            existing,
            incoming,
            previous_chunk_index=0,
            next_chunk_index=1,
            overlap_start_seconds=9,
            overlap_end_seconds=11,
            incoming_warning=None,
        )

        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["incoming_head"][0]["decision"], "trimmed_duplicate_prefix")
        self.assertEqual(merged[1]["transcript"], "nomi e cognomi e zone")
        self.assertGreater(merged[1]["start"], incoming[0]["start"])

    def test_merge_trim_keeps_matching_word_slice(self) -> None:
        existing = [
            self._segment(0, 10, "Io credo non ho"),
        ]
        incoming = [
            self._word_segment(9, 15, "Io credo non ho paura adesso"),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            existing,
            incoming,
            previous_chunk_index=0,
            next_chunk_index=1,
            overlap_start_seconds=9,
            overlap_end_seconds=11,
            incoming_warning=None,
        )

        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["incoming_head"][0]["decision"], "trimmed_duplicate_prefix")
        self.assertEqual(merged[1]["transcript"], "paura adesso")
        self.assertEqual([item["word"] for item in merged[1]["words"]], ["paura", "adesso"])

    def test_merge_trims_partial_duplicate_suffix(self) -> None:
        existing = [
            self._segment(0, 10, "del nostro investimento quindi ha rimesso in ordine l'area"),
        ]
        incoming = [
            self._segment(9, 15, "prima parole nuove del nostro investimento quindi ha rimesso in ordine l'area"),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            existing,
            incoming,
            previous_chunk_index=0,
            next_chunk_index=1,
            overlap_start_seconds=9,
            overlap_end_seconds=15,
            incoming_warning=None,
        )

        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["incoming_head"][0]["decision"], "trimmed_duplicate_suffix")
        self.assertEqual(merged[1]["transcript"], "prima parole nuove")
        self.assertLess(merged[1]["end"], incoming[0]["end"])

    def test_merge_regression_drops_split_boundary_duplicates(self) -> None:
        existing = [
            self._segment(
                5305.91,
                5312.91,
                (
                    "una condizione concreta in cui eventi esterni non imputabili alla gestione sportiva "
                    "hanno inciso sull'equilibrio economico della societa."
                ),
            ),
            self._segment(
                5313.53,
                5317.97,
                "I lavori risultano programmati e saranno oggetto di aggiornamento da parte degli uffici tecnici",
            ),
            self._segment(5317.97, 5321.03, "competenti per quanto riguarda tempi e stato di avanzamento."),
        ]
        incoming = [
            self._segment(5310.56, 5313.08, "hanno inciso sull'equilibrio economico della societa."),
            self._segment(
                5313.08,
                5318.46,
                "I lavori risultano programmati e saranno oggetto di aggiornamento da parte degli uffici tecnici competenti",
            ),
            self._segment(5318.46, 5321.04, "per quanto riguarda tempi e stato di avanzamento."),
            self._segment(5321.04, 5325.38, "Nel frattempo ricordo che oltre a questo investimento che andiamo a fare"),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            existing,
            incoming,
            previous_chunk_index=2,
            next_chunk_index=3,
            overlap_start_seconds=5310,
            overlap_end_seconds=5340,
            incoming_warning=None,
        )

        self.assertIsNotNone(audit)
        assert audit is not None
        merged_text = " ".join(item["transcript"] for item in merged)
        self.assertEqual(merged_text.count("hanno inciso sull'equilibrio economico della societa"), 1)
        self.assertEqual(merged_text.count("I lavori risultano programmati"), 1)
        self.assertEqual(audit["counts"]["dropped_overlap_duplicates"], 3)

    def test_merge_with_audit_first_chunk_has_no_audit(self) -> None:
        incoming = [
            self._segment(0, 5, "Inizio"),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            [],
            incoming,
            previous_chunk_index=None,
            next_chunk_index=0,
            overlap_start_seconds=0,
            overlap_end_seconds=None,
            incoming_warning=None,
        )

        self.assertEqual(merged, incoming)
        self.assertIsNone(audit)

    def test_render_stitch_audit_markdown_contains_decisions(self) -> None:
        _merged, audit = merge_chunk_segments_with_audit(
            [self._segment(3580, 3590, "Ciao mondo.")],
            [
                self._segment(3575, 3585, "ciao mondo"),
                self._segment(3591, 3593, "Frase nuova"),
            ],
            previous_chunk_index=2,
            next_chunk_index=3,
            overlap_start_seconds=3570,
            overlap_end_seconds=3600,
            incoming_warning={"chunk": 3, "type": "repetition_retry_succeeded", "phrase": "ciao mondo"},
        )
        assert audit is not None

        markdown = render_stitch_audit_markdown(audit)

        self.assertIn("Chunk 0002 -> 0003", markdown)
        self.assertIn("00:59:30.000 -> 01:00:00.000", markdown)
        self.assertIn("dropped_exact_duplicate", markdown)
        self.assertIn("kept_low_confidence_overlap", markdown)
        self.assertIn("Frase nuova", markdown)
        self.assertIn("repetition_retry_succeeded", markdown)

    def test_safe_zone_keeps_middle_segments_by_midpoint(self) -> None:
        existing = [self._segment(0, 55, "Chunk precedente")]
        incoming = [
            self._segment(50, 54, "Bordo iniziale"),
            self._segment(55, 65, "Centro utile"),
            self._segment(114, 120, "Bordo finale"),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            existing,
            incoming,
            previous_chunk_index=0,
            next_chunk_index=1,
            overlap_start_seconds=50,
            overlap_end_seconds=60,
            overlap_seconds=10,
            chunk_start_seconds=50,
            chunk_end_seconds=120,
            is_first_chunk=False,
            is_last_chunk=False,
            incoming_warning=None,
            stitch_method="safe_zone",
        )

        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual([item["transcript"] for item in merged], ["Chunk precedente", "Centro utile"])
        self.assertEqual(audit["method"], "safe_zone")
        self.assertEqual(audit["safe_start"], 55)
        self.assertEqual(audit["safe_end"], 115)
        self.assertEqual(audit["counts"]["kept"], 1)
        self.assertEqual(audit["counts"]["dropped_safe_zone"], 2)

    def test_safe_zone_first_chunk_drops_trailing_edge(self) -> None:
        incoming = [
            self._segment(0, 10, "Inizio"),
            self._segment(114, 120, "Bordo finale"),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            [],
            incoming,
            previous_chunk_index=None,
            next_chunk_index=0,
            overlap_start_seconds=0,
            overlap_end_seconds=None,
            overlap_seconds=10,
            chunk_start_seconds=0,
            chunk_end_seconds=120,
            is_first_chunk=True,
            is_last_chunk=False,
            incoming_warning=None,
            stitch_method="safe_zone",
        )

        self.assertIsNone(audit)
        self.assertEqual([item["transcript"] for item in merged], ["Inizio"])

    def test_safe_zone_last_chunk_keeps_until_chunk_end(self) -> None:
        existing = [self._segment(0, 55, "Chunk precedente")]
        incoming = [
            self._segment(55, 65, "Centro utile"),
            self._segment(116, 120, "Fine utile"),
        ]

        merged, audit = merge_chunk_segments_with_audit(
            existing,
            incoming,
            previous_chunk_index=0,
            next_chunk_index=1,
            overlap_start_seconds=50,
            overlap_end_seconds=60,
            overlap_seconds=10,
            chunk_start_seconds=50,
            chunk_end_seconds=120,
            is_first_chunk=False,
            is_last_chunk=True,
            incoming_warning=None,
            stitch_method="safe_zone",
        )

        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual([item["transcript"] for item in merged], ["Chunk precedente", "Centro utile", "Fine utile"])
        self.assertEqual(audit["safe_end"], 120)

    def test_format_timestamp(self) -> None:
        self.assertEqual(format_timestamp(3661.234), "01:01:01.234")

    def test_detect_repetition_catches_phrase_loop(self) -> None:
        segments = [{"transcript": "non c'e nessun problema"} for _ in range(5)]

        report = detect_repetition(segments)

        self.assertTrue(report.detected)
        self.assertEqual(report.first_bad_index, 0)

    def test_detect_repetition_ignores_short_words(self) -> None:
        segments = [{"transcript": "si"} for _ in range(8)]

        report = detect_repetition(segments)

        self.assertFalse(report.detected)

    def test_extract_segments_populates_words_from_json_full_tokens(self) -> None:
        raw = {
            "transcription": [
                {
                    "offsets": {"from": 1000, "to": 5000},
                    "text": " Buongiorno a tutti. L'assessore arriva.",
                    "tokens": [
                        {"text": " Bu", "offsets": {"from": 1000, "to": 1100}},
                        {"text": "ong", "offsets": {"from": 1100, "to": 1250}},
                        {"text": "ior", "offsets": {"from": 1250, "to": 1400}},
                        {"text": "no", "offsets": {"from": 1400, "to": 1500}},
                        {"text": " a", "offsets": {"from": 1600, "to": 1700}},
                        {"text": " tutti", "offsets": {"from": 1800, "to": 2200}},
                        {"text": ".", "offsets": {"from": 2200, "to": 2300}},
                        {"text": " L", "offsets": {"from": 2500, "to": 2600}},
                        {"text": "'", "offsets": {"from": 2600, "to": 2650}},
                        {"text": "ass", "offsets": {"from": 2650, "to": 2800}},
                        {"text": "essore", "offsets": {"from": 2800, "to": 3200}},
                        {"text": " arriva", "offsets": {"from": 3500, "to": 4300}},
                        {"text": "[_TT_1195]", "offsets": {"from": 4300, "to": 4300}},
                    ],
                }
            ]
        }

        segments = _extract_segments(raw, offset_seconds=10)

        self.assertEqual(segments[0]["start"], 11)
        self.assertEqual(segments[0]["end"], 15)
        self.assertEqual(
            segments[0]["words"],
            [
                {"word": "Buongiorno", "start": 11, "end": 11.5},
                {"word": "a", "start": 11.6, "end": 11.7},
                {"word": "tutti", "start": 11.8, "end": 12.2},
                {"word": "L'assessore", "start": 12.5, "end": 13.2},
                {"word": "arriva", "start": 13.5, "end": 14.3},
            ],
        )

    def test_extract_segments_clips_words_to_chunk_window(self) -> None:
        raw = {
            "transcription": [
                {
                    "offsets": {"from": 0, "to": 4000},
                    "text": " Prima dentro dopo",
                    "tokens": [
                        {"text": " Prima", "offsets": {"from": 0, "to": 500}},
                        {"text": " dentro", "offsets": {"from": 1000, "to": 2000}},
                        {"text": " dopo", "offsets": {"from": 3000, "to": 4000}},
                    ],
                }
            ]
        }

        segments = _extract_segments(raw, offset_seconds=100, clamp_start=101, clamp_end=102.4)

        self.assertEqual(segments[0]["start"], 101)
        self.assertEqual(segments[0]["end"], 102.4)
        self.assertEqual(segments[0]["words"], [{"word": "dentro", "start": 101, "end": 102}])

    def test_extract_segments_handles_vad_relative_token_offsets(self) -> None:
        raw = {
            "transcription": [
                {
                    "offsets": {"from": 90000, "to": 93000},
                    "text": " Buongiorno",
                    "tokens": [
                        {"text": " Bu", "offsets": {"from": 500, "to": 600}},
                        {"text": "ong", "offsets": {"from": 600, "to": 750}},
                        {"text": "iorno", "offsets": {"from": 750, "to": 1100}},
                    ],
                }
            ]
        }

        segments = _extract_segments(raw, offset_seconds=1000)

        self.assertEqual(segments[0]["start"], 1090)
        self.assertEqual(segments[0]["end"], 1093)
        self.assertEqual(segments[0]["words"], [{"word": "Buongiorno", "start": 1090.5, "end": 1091.1}])


if __name__ == "__main__":
    unittest.main()
