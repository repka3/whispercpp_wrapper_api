# Wrapper Plan: Diagnose Hard Chunking and Add Silence-Aligned Splitting

## Summary

The golden benchmark app has shown that fixed timestamp chunks add significant WER even with 30s overlap. This is not only a stitch-boundary problem: actual job artifacts show many added errors before and after the boundary zone, meaning chunked decoding itself changes the transcript. The wrapper should own the fix because it owns chunk windows, VAD, chunk extraction, raw chunk artifacts, stitching, and repetition guard behavior.

## Findings So Far

- Golden benchmark results with `ggml-large-v3-turbo.bin`, 30s overlap:
  - `120s` chunks: median added WER about `+6.1pp`; every sample worse.
  - `300s` chunks: median added WER about `+3.5pp`; still mostly worse.
  - `600s` chunks: median added WER about `+2.8pp` to `+3.1pp`; still not clean.
- Longer chunks reduce damage, so boundary count matters, but even one hard boundary still adds meaningful error.
- Actual wrapper job inspection for `600s` chunks showed errors are not concentrated at the boundary:
  - `r05780`: approx `23` errors before boundary zone, `6` near boundary, `28` after.
  - `r02098`: approx `29` before, `8` near boundary, `25` after.
- Current stitch methods are segment-level heuristics:
  - `fuzzy`: exact/token-overlap duplicate drop and prefix/suffix trim.
  - `safe_zone`: keeps/drops by segment midpoint in the safe zone.
  - Neither performs full word-level optimal alignment across the overlap.
- A 30s overlap cannot fix a chunk that decoded differently in its non-overlap middle.
- Local `whisper.cpp` has `build/bin/whisper-vad-speech-segments` and `models/ggml-silero-v6.2.0.bin`, so the silence-aligned split can use existing Silero VAD tooling.

## Key Changes To Try In The Wrapper

- Add diagnostics before changing production behavior:
  - Persist a chunk manifest JSON with window start/end, duration, overlap, chunk segment count, warnings, retry status, and source path.
  - Extend stitch debug JSON enough to inspect each boundary without opening logs.
  - Add an offline/debug helper or test utility that compares raw chunk text against the no-chunk baseline by approximate timestamp range, separating chunk decode error from stitch error.
  - Add an oracle stitch test: split an already-correct transcript/segment list into artificial chunks, stitch it, and prove the stitcher itself does not create errors on clean inputs.

- Add `chunking_mode` to wrapper API/config:
  - Supported values: `fixed` and `vad_silence`.
  - Default: `fixed`, preserving current behavior.
  - In `vad_silence`, `chunk_seconds` means target chunk length, not a hard cut.
  - Add optional `silence_search_radius_seconds`; default to `min(chunk_seconds / 2, 900)`.
  - Add optional `silence_min_duration_ms`; default `700`.
  - If `chunking_mode=vad_silence` and `chunk_overlap_seconds` is omitted, default effective overlap to `0`.
  - If `chunk_overlap_seconds` is explicitly set, honor it as boundary padding/overlap for experiments.

- Implement VAD-silence window building:
  - Run `whisper-vad-speech-segments` using the configured Silero VAD model.
  - Parse speech segments and infer silence gaps.
  - For each target cut around `N * chunk_seconds`, search within `target +/- silence_search_radius_seconds`.
  - Choose the silence midpoint closest to the target, requiring at least `silence_min_duration_ms`.
  - If no valid silence exists, fall back to the fixed target cut and record `fallback: true` in the manifest.
  - Build non-overlapping windows from cut to cut for the default silence mode.
  - Include cut decisions in result metadata: target, selected cut, distance, silence duration, fallback reason.

- Keep the golden benchmark app as evaluator only:
  - It should continue running `chunk_seconds=0` baseline and measuring added WER.
  - Later, add a benchmark option for `chunking_mode=vad_silence`.
  - Do not move VAD/window/stitch logic into the golden app.

## Test Plan

- Wrapper unit tests:
  - `build_chunk_windows` existing fixed behavior remains unchanged.
  - VAD output parser handles `whisper-vad-speech-segments` text output and normalizes time units correctly.
  - Silence window builder picks the closest valid silence inside the search window.
  - Silence window builder falls back to fixed cuts when no silence is found.
  - Short audio produces one window.
  - Explicit overlap in `vad_silence` mode is honored; omitted overlap becomes `0`.
  - Oracle stitch produces unchanged text on artificial clean chunk splits.
  - Existing `fuzzy`, `safe_zone`, and `chunk_seconds=0` tests keep passing.

- Wrapper command:
  - Run from `whispercpp_wrapper_api` with `.venv/bin/python -m unittest discover -v`.

- Benchmark acceptance:
  - Re-run golden benchmark for fixed `600s` and `vad_silence` target `600s`.
  - Then test production-like target `1800s` on longer meetings.
  - Success is not perfect WER; success is a clear reduction in median added WER versus fixed hard cuts, with fewer boundary-local errors and no broad regression across samples.

## Assumptions

- The wrapper project owns all production chunking and stitching behavior.
- The golden benchmark project owns evaluation, WER comparison, screenshots, and result interpretation.
- The first goal is diagnosis and safer cut placement, not immediately building a complex LLM/word-level stitcher.
- Silence-aligned chunks should be tested with effective overlap `0` first, because the goal is to make concatenation nearly sufficient.
