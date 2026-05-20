#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.golden_debug import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_STITCH_METHODS,
    DEFAULT_VAD_CUT_THRESHOLD,
    DEFAULT_VAD_THRESHOLD,
    run_golden_debug,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden transcription cut diagnostics.")
    parser.add_argument("--golden-dir", type=Path, default=Path("golden"))
    parser.add_argument("--out-dir", type=Path, default=Path("golden_runs"))
    parser.add_argument("--model", default="ggml-large-v3-turbo.bin")
    parser.add_argument("--language", default="it")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--vad-threshold", type=float, default=DEFAULT_VAD_THRESHOLD)
    parser.add_argument("--vad-cut-threshold", type=float, default=DEFAULT_VAD_CUT_THRESHOLD)
    parser.add_argument("--vad-max-speech-duration-s", type=int, default=3600)
    parser.add_argument("--vad-min-silence-duration-ms", type=int, default=2000)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=400)
    parser.add_argument("--chunk-seconds", type=int, nargs="+", default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--chunk-overlap-seconds", type=int, default=30)
    parser.add_argument("--stitch-methods", nargs="+", default=DEFAULT_STITCH_METHODS)
    parser.add_argument("--only", action="append", help="Run cases whose filename contains this text.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--planner-only", action="store_true", help="Only run VAD/cut planning diagnostics.")
    args = parser.parse_args()

    settings = get_settings()
    run_dir = run_golden_debug(
        golden_dir=args.golden_dir,
        out_dir=args.out_dir,
        settings=settings,
        model=args.model,
        language=args.language,
        beam_size=args.beam_size,
        best_of=args.best_of,
        vad_threshold=args.vad_threshold,
        vad_cut_threshold=args.vad_cut_threshold,
        vad_max_speech_duration_s=args.vad_max_speech_duration_s,
        vad_min_silence_duration_ms=args.vad_min_silence_duration_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        chunk_seconds_values=args.chunk_seconds,
        chunk_overlap_seconds=args.chunk_overlap_seconds,
        stitch_methods=args.stitch_methods,
        only=args.only,
        limit=args.limit,
        planner_only=args.planner_only,
    )
    print(f"Wrote golden debug run to {run_dir}")
    print(f"Summary: {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
