import tempfile
import unittest
from pathlib import Path

from app.golden_debug import compute_wer, discover_golden_pairs, normalize_for_wer


class GoldenDebugTests(unittest.TestCase):
    def test_discover_golden_pairs_matches_audio_to_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "meeting-a.mp3").write_bytes(b"audio")
            (root / "meeting-a.txt").write_text("reference", encoding="utf-8")
            (root / "meeting-b.mp3").write_bytes(b"audio")

            pairs = discover_golden_pairs(root)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].name, "meeting-a")

    def test_discover_golden_pairs_filters_by_filename_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "consiglio-alpha.mp3").write_bytes(b"audio")
            (root / "consiglio-alpha.txt").write_text("reference", encoding="utf-8")
            (root / "riunione-beta.mp3").write_bytes(b"audio")
            (root / "riunione-beta.txt").write_text("reference", encoding="utf-8")

            pairs = discover_golden_pairs(root, only=["beta"])

        self.assertEqual([item.name for item in pairs], ["riunione-beta"])

    def test_normalize_for_wer_keeps_italian_apostrophes(self) -> None:
        self.assertEqual(
            normalize_for_wer("L'assessore, però, parla."),
            "l'assessore però parla",
        )

    def test_compute_wer_reports_substitution_deletion_and_insertion(self) -> None:
        result = compute_wer("uno due tre quattro", "uno due diverso quattro extra")

        self.assertEqual(result.reference_words, 4)
        self.assertEqual(result.hypothesis_words, 5)
        self.assertEqual(result.substitutions, 1)
        self.assertEqual(result.deletions, 0)
        self.assertEqual(result.insertions, 1)
        self.assertEqual(result.wer, 0.5)
        self.assertIn({"op": "substitute", "ref": "tre", "hyp": "diverso"}, result.operations)

    def test_compute_wer_reports_deletion(self) -> None:
        result = compute_wer("uno due tre", "uno tre")

        self.assertEqual(result.substitutions, 0)
        self.assertEqual(result.deletions, 1)
        self.assertEqual(result.insertions, 0)
        self.assertEqual(result.wer, 1 / 3)


if __name__ == "__main__":
    unittest.main()
