import unittest

from app.whisper import (
    build_chunk_windows,
    detect_repetition,
    format_timestamp,
    merge_chunk_segments,
    merge_chunk_segments_with_audit,
    render_stitch_audit_markdown,
)


class WhisperHelperTests(unittest.TestCase):
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

    def test_merge_chunk_segments_dedupes_overlap(self) -> None:
        existing = [
            {"start": 0, "end": 10, "transcript": "Ciao mondo.", "words": []},
        ]
        incoming = [
            {"start": 9, "end": 11, "transcript": "ciao mondo", "words": []},
            {"start": 12, "end": 13, "transcript": "Nuovo testo", "words": []},
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
            {"start": 0, "end": 10, "transcript": "Ciao mondo.", "words": []},
        ]
        incoming = [
            {"start": 9, "end": 11, "transcript": "ciao mondo", "words": []},
            {"start": 12, "end": 13, "transcript": "Nuovo testo", "words": []},
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
        self.assertEqual(audit["incoming_head"][0]["decision"], "dropped_duplicate")

    def test_merge_with_audit_keeps_disagreement_inside_overlap(self) -> None:
        existing = [
            {"start": 0, "end": 10, "transcript": "Primo testo", "words": []},
        ]
        incoming = [
            {"start": 9, "end": 11, "transcript": "Testo diverso", "words": []},
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
        self.assertEqual([item["transcript"] for item in merged], ["Primo testo", "Testo diverso"])
        self.assertEqual(audit["incoming_head"][0]["decision"], "kept")
        self.assertEqual(audit["counts"]["kept_overlap"], 1)

    def test_merge_with_audit_first_chunk_has_no_audit(self) -> None:
        incoming = [
            {"start": 0, "end": 5, "transcript": "Inizio", "words": []},
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
            [{"start": 3580, "end": 3590, "transcript": "Ciao mondo.", "words": []}],
            [
                {"start": 3575, "end": 3585, "transcript": "ciao mondo", "words": []},
                {"start": 3591, "end": 3593, "transcript": "Frase nuova", "words": []},
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
        self.assertIn("dropped_duplicate", markdown)
        self.assertIn("kept", markdown)
        self.assertIn("Frase nuova", markdown)
        self.assertIn("repetition_retry_succeeded", markdown)

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


if __name__ == "__main__":
    unittest.main()
