import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .config import Settings
from .whisper import (
    build_chunk_windows,
    build_mixed_aligned_chunk_windows,
    parse_vad_speech_segments,
    probe_duration_seconds,
    run_transcription,
    run_vad_speech_segments,
    _infer_silence_gaps,
)


DEFAULT_CHUNK_SECONDS = [120, 300]
DEFAULT_STITCH_METHODS = ["fuzzy", "safe_zone", "word_align", "center_align"]
DEFAULT_VAD_THRESHOLD = 0.01
DEFAULT_VAD_CUT_THRESHOLD = 0.5
EXACT_WER_CELL_LIMIT = 10_000_000
SUMMARY_COLUMNS = [
    "case",
    "mode",
    "stitch_method",
    "duration_seconds",
    "chunk_seconds",
    "requested_overlap_seconds",
    "effective_overlap_seconds",
    "strategy",
    "chunk_count",
    "fallback_reason",
    "wer",
    "wer_delta_vs_full",
    "substitutions",
    "deletions",
    "insertions",
    "reference_words",
    "hypothesis_words",
    "elapsed_seconds",
    "rtf",
    "result_path",
    "debug_path",
]


@dataclass(frozen=True)
class GoldenPair:
    name: str
    audio_path: Path
    reference_path: Path


@dataclass(frozen=True)
class WerResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    hypothesis_words: int
    operations: list[dict[str, str]]


def discover_golden_pairs(golden_dir: Path, only: list[str] | None = None) -> list[GoldenPair]:
    filters = [item.lower() for item in (only or []) if item]
    pairs: list[GoldenPair] = []
    for audio_path in sorted(golden_dir.glob("*.mp3")):
        reference_path = audio_path.with_suffix(".txt")
        if not reference_path.exists():
            continue
        if filters and not any(item in audio_path.stem.lower() for item in filters):
            continue
        pairs.append(GoldenPair(name=audio_path.stem, audio_path=audio_path, reference_path=reference_path))
    return pairs


def normalize_for_wer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(reference: str, hypothesis: str) -> WerResult:
    ref_words = normalize_for_wer(reference).split()
    hyp_words = normalize_for_wer(hypothesis).split()
    ref_len = len(ref_words)
    hyp_len = len(hyp_words)
    if ref_len * hyp_len > EXACT_WER_CELL_LIMIT:
        return _compute_wer_blockwise(ref_words, hyp_words)

    dp: list[list[tuple[int, int, int, int, str | None]]] = [
        [(0, 0, 0, 0, None) for _ in range(hyp_len + 1)] for _ in range(ref_len + 1)
    ]

    for i in range(1, ref_len + 1):
        cost, sub, delete, insert, _op = dp[i - 1][0]
        dp[i][0] = (cost + 1, sub, delete + 1, insert, "delete")
    for j in range(1, hyp_len + 1):
        cost, sub, delete, insert, _op = dp[0][j - 1]
        dp[0][j] = (cost + 1, sub, delete, insert + 1, "insert")

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost, sub, delete, insert, _op = dp[i - 1][j - 1]
                dp[i][j] = (cost, sub, delete, insert, "equal")
                continue

            sub_cost, sub_s, sub_d, sub_i, _op = dp[i - 1][j - 1]
            del_cost, del_s, del_d, del_i, _op = dp[i - 1][j]
            ins_cost, ins_s, ins_d, ins_i, _op = dp[i][j - 1]
            candidates = [
                (sub_cost + 1, sub_s + 1, sub_d, sub_i, "substitute"),
                (del_cost + 1, del_s, del_d + 1, del_i, "delete"),
                (ins_cost + 1, ins_s, ins_d, ins_i + 1, "insert"),
            ]
            dp[i][j] = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))

    operations: list[dict[str, str]] = []
    i = ref_len
    j = hyp_len
    while i > 0 or j > 0:
        op = dp[i][j][4]
        if op == "equal":
            operations.append({"op": "equal", "ref": ref_words[i - 1], "hyp": hyp_words[j - 1]})
            i -= 1
            j -= 1
        elif op == "substitute":
            operations.append({"op": "substitute", "ref": ref_words[i - 1], "hyp": hyp_words[j - 1]})
            i -= 1
            j -= 1
        elif op == "delete":
            operations.append({"op": "delete", "ref": ref_words[i - 1], "hyp": ""})
            i -= 1
        elif op == "insert":
            operations.append({"op": "insert", "ref": "", "hyp": hyp_words[j - 1]})
            j -= 1
        else:
            break
    operations.reverse()

    cost, substitutions, deletions, insertions, _op = dp[ref_len][hyp_len]
    wer = 0.0 if ref_len == 0 and cost == 0 else (1.0 if ref_len == 0 else cost / ref_len)
    return WerResult(
        wer=wer,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=ref_len,
        hypothesis_words=hyp_len,
        operations=operations,
    )


def _compute_wer_blockwise(ref_words: list[str], hyp_words: list[str]) -> WerResult:
    matcher = SequenceMatcher(None, ref_words, hyp_words, autojunk=True)
    substitutions = 0
    deletions = 0
    insertions = 0
    operations: list[dict[str, str]] = []

    for op, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        ref_block = ref_words[ref_start:ref_end]
        hyp_block = hyp_words[hyp_start:hyp_end]
        if op == "equal":
            continue
        if op == "delete":
            deletions += len(ref_block)
            continue
        if op == "insert":
            insertions += len(hyp_block)
            continue

        sub, delete, insert = _edit_counts_for_block(ref_block, hyp_block)
        substitutions += sub
        deletions += delete
        insertions += insert

    cost = substitutions + deletions + insertions
    ref_len = len(ref_words)
    wer = 0.0 if ref_len == 0 and cost == 0 else (1.0 if ref_len == 0 else cost / ref_len)
    return WerResult(
        wer=wer,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=ref_len,
        hypothesis_words=len(hyp_words),
        operations=operations,
    )


def _edit_counts_for_block(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int, int]:
    if not ref_words:
        return 0, 0, len(hyp_words)
    if not hyp_words:
        return 0, len(ref_words), 0
    if len(ref_words) * len(hyp_words) > EXACT_WER_CELL_LIMIT:
        substitutions = min(len(ref_words), len(hyp_words))
        deletions = max(len(ref_words) - len(hyp_words), 0)
        insertions = max(len(hyp_words) - len(ref_words), 0)
        return substitutions, deletions, insertions

    previous = [(j, 0, 0, j) for j in range(len(hyp_words) + 1)]
    for i, ref_word in enumerate(ref_words, start=1):
        current: list[tuple[int, int, int, int]] = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hyp_words, start=1):
            if ref_word == hyp_word:
                current.append(previous[j - 1])
                continue
            sub_cost, sub_s, sub_d, sub_i = previous[j - 1]
            del_cost, del_s, del_d, del_i = previous[j]
            ins_cost, ins_s, ins_d, ins_i = current[j - 1]
            current.append(
                min(
                    [
                        (sub_cost + 1, sub_s + 1, sub_d, sub_i),
                        (del_cost + 1, del_s, del_d + 1, del_i),
                        (ins_cost + 1, ins_s, ins_d, ins_i + 1),
                    ],
                    key=lambda item: (item[0], item[1], item[2], item[3]),
                )
            )
        previous = current
    _cost, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions


def run_golden_debug(
    *,
    golden_dir: Path,
    out_dir: Path,
    settings: Settings,
    model: str,
    language: str,
    beam_size: int,
    best_of: int,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    vad_cut_threshold: float = DEFAULT_VAD_CUT_THRESHOLD,
    vad_max_speech_duration_s: int = 3600,
    vad_min_silence_duration_ms: int = 2000,
    vad_speech_pad_ms: int = 400,
    chunk_seconds_values: list[int] | None = None,
    chunk_overlap_seconds: int = 30,
    stitch_methods: list[str] | None = None,
    only: list[str] | None = None,
    limit: int | None = None,
    planner_only: bool = False,
) -> Path:
    pairs = discover_golden_pairs(golden_dir, only=only)
    if limit is not None:
        pairs = pairs[: max(limit, 0)]
    chunk_seconds_values = chunk_seconds_values or list(DEFAULT_CHUNK_SECONDS)
    stitch_methods = stitch_methods or list(DEFAULT_STITCH_METHODS)
    primary_stitch_method = stitch_methods[0]

    run_dir = out_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "config.json",
        {
            "golden_dir": str(golden_dir),
            "model": model,
            "language": language,
            "beam_size": beam_size,
            "best_of": best_of,
            "vad_threshold": vad_threshold,
            "vad_cut_threshold": vad_cut_threshold,
            "vad_max_speech_duration_s": vad_max_speech_duration_s,
            "vad_min_silence_duration_ms": vad_min_silence_duration_ms,
            "vad_speech_pad_ms": vad_speech_pad_ms,
            "chunk_seconds": chunk_seconds_values,
            "chunk_overlap_seconds": chunk_overlap_seconds,
            "stitch_methods": stitch_methods,
            "planner_only": planner_only,
        },
    )

    rows: list[dict[str, Any]] = []
    model_path = None if planner_only else settings.resolve_model(model)
    for pair in pairs:
        case_dir = run_dir / _safe_slug(pair.name)
        case_dir.mkdir(parents=True, exist_ok=True)
        reference_text = pair.reference_path.read_text(encoding="utf-8")
        duration = probe_duration_seconds(pair.audio_path)
        full_wer: WerResult | None = None
        full_text: str | None = None

        if not planner_only and model_path is not None:
            full_result = _run_case_transcription(
                pair=pair,
                job_dir=case_dir / "full",
                settings=settings,
                model_path=model_path,
                language=language,
                beam_size=beam_size,
                best_of=best_of,
                vad_threshold=vad_threshold,
                vad_cut_threshold=vad_cut_threshold,
                vad_max_speech_duration_s=vad_max_speech_duration_s,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
                chunk_seconds=0,
                chunk_overlap_seconds=0,
                stitch_method=None,
                stitch_methods=None,
            )
            full_text = full_result.get("text", "")
            full_wer = _write_text_evaluation(
                case_dir=case_dir / "full",
                reference_text=reference_text,
                hypothesis_text=full_text,
                result=full_result,
            )
            rows.append(
                _summary_row(
                    pair=pair,
                    mode="full",
                    stitch_method="",
                    duration=duration,
                    chunk_seconds=0,
                    requested_overlap_seconds=0,
                    result=full_result,
                    wer=full_wer,
                    full_wer=full_wer,
                    debug_path=case_dir / "full" / "evaluation.json",
                )
            )

        for chunk_seconds in chunk_seconds_values:
            planner = build_planner_diagnostics(
                pair.audio_path,
                settings=settings,
                chunk_seconds=chunk_seconds,
                chunk_overlap_seconds=chunk_overlap_seconds,
                vad_threshold=vad_threshold,
                vad_cut_threshold=vad_cut_threshold,
                vad_max_speech_duration_s=vad_max_speech_duration_s,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
                audio_duration_seconds=duration,
            )
            planner_dir = case_dir / f"chunk_{chunk_seconds}" / "planner"
            planner_dir.mkdir(parents=True, exist_ok=True)
            _write_json(planner_dir / "planner.json", planner)

            if planner_only or model_path is None:
                rows.append(
                    _planner_summary_row(
                        pair=pair,
                        duration=duration,
                        chunk_seconds=chunk_seconds,
                        requested_overlap_seconds=chunk_overlap_seconds,
                        planner=planner,
                        debug_path=planner_dir / "planner.json",
                    )
                )
                continue

            chunk_result = _run_case_transcription(
                pair=pair,
                job_dir=case_dir / f"chunk_{chunk_seconds}",
                settings=settings,
                model_path=model_path,
                language=language,
                beam_size=beam_size,
                best_of=best_of,
                vad_threshold=vad_threshold,
                vad_cut_threshold=vad_cut_threshold,
                vad_max_speech_duration_s=vad_max_speech_duration_s,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
                chunk_seconds=chunk_seconds,
                chunk_overlap_seconds=chunk_overlap_seconds,
                stitch_method=primary_stitch_method,
                stitch_methods=stitch_methods,
            )
            variants = chunk_result.get("stitch_variants") or {}
            if variants:
                for method, variant in variants.items():
                    method_dir = case_dir / f"chunk_{chunk_seconds}" / f"variant_{method}"
                    method_dir.mkdir(parents=True, exist_ok=True)
                    method_wer = _write_text_evaluation(
                        case_dir=method_dir,
                        reference_text=reference_text,
                        hypothesis_text=variant.get("text", ""),
                        result=variant,
                        baseline_text=full_text,
                    )
                    rows.append(
                        _summary_row(
                            pair=pair,
                            mode=f"chunk_{chunk_seconds}",
                            stitch_method=method,
                            duration=duration,
                            chunk_seconds=chunk_seconds,
                            requested_overlap_seconds=chunk_overlap_seconds,
                            result=variant,
                            wer=method_wer,
                            full_wer=full_wer,
                            debug_path=method_dir / "evaluation.json",
                        )
                    )
            else:
                chunk_wer = _write_text_evaluation(
                    case_dir=case_dir / f"chunk_{chunk_seconds}",
                    reference_text=reference_text,
                    hypothesis_text=chunk_result.get("text", ""),
                    result=chunk_result,
                    baseline_text=full_text,
                )
                rows.append(
                    _summary_row(
                        pair=pair,
                        mode=f"chunk_{chunk_seconds}",
                        stitch_method=primary_stitch_method,
                        duration=duration,
                        chunk_seconds=chunk_seconds,
                        requested_overlap_seconds=chunk_overlap_seconds,
                        result=chunk_result,
                        wer=chunk_wer,
                        full_wer=full_wer,
                        debug_path=case_dir / f"chunk_{chunk_seconds}" / "evaluation.json",
                    )
                )

    _write_summary_csv(run_dir / "summary.csv", rows)
    _write_summary_markdown(run_dir / "summary.md", rows)
    return run_dir


def build_planner_diagnostics(
    input_path: Path,
    *,
    settings: Settings,
    chunk_seconds: int,
    chunk_overlap_seconds: int,
    vad_threshold: float,
    vad_cut_threshold: float | None = None,
    vad_max_speech_duration_s: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    audio_duration_seconds: float | None = None,
) -> dict[str, Any]:
    duration = probe_duration_seconds(input_path) if audio_duration_seconds is None else audio_duration_seconds
    fixed_windows = build_chunk_windows(
        audio_duration_seconds=duration,
        chunk_seconds=chunk_seconds,
        overlap_seconds=chunk_overlap_seconds,
    )
    speech_segments = []
    silence_gaps = []
    cut_decisions = []
    strategy = "fixed"
    effective_overlap = chunk_overlap_seconds
    warnings: list[dict[str, Any]] = []
    windows = fixed_windows
    vad_output = ""

    if len(fixed_windows) > 1:
        try:
            vad_output = run_vad_speech_segments(
                input_path=input_path,
                settings=settings,
                vad_threshold=vad_threshold if vad_cut_threshold is None else vad_cut_threshold,
                vad_max_speech_duration_s=vad_max_speech_duration_s,
                vad_min_silence_duration_ms=vad_min_silence_duration_ms,
                vad_speech_pad_ms=vad_speech_pad_ms,
            )
            parsed_segments = parse_vad_speech_segments(vad_output)
            speech_segments = [asdict(item) for item in parsed_segments]
            raw_gaps = _infer_silence_gaps(
                speech_segments=parsed_segments,
                audio_duration_seconds=duration,
                min_silence_seconds=vad_min_silence_duration_ms / 1000.0,
            )
            silence_gaps = [
                {"start_seconds": round(start, 3), "end_seconds": round(end, 3), "duration_seconds": round(end - start, 3)}
                for start, end in raw_gaps
            ]
            windows, decisions = build_mixed_aligned_chunk_windows(
                audio_duration_seconds=duration,
                chunk_seconds=chunk_seconds,
                speech_segments=parsed_segments,
                silence_min_duration_ms=vad_min_silence_duration_ms,
                overlap_seconds=chunk_overlap_seconds,
            )
            cut_decisions = [item.as_dict() for item in decisions]
            strategy = "mixed" if any(item.cut_type == "hard_fallback" for item in decisions) else "vad_silence"
            effective_overlap = chunk_overlap_seconds
        except (OSError, ValueError) as exc:
            strategy = "fixed_fallback"
            effective_overlap = chunk_overlap_seconds
            windows = fixed_windows
            warnings.append(
                {
                    "type": "chunking_fixed_fallback",
                    "reason": str(exc),
                    "message": f"Silence-aligned chunk planning failed; using fixed hard cuts. Reason: {exc}",
                }
            )

    return {
        "input_path": str(input_path),
        "audio_duration_seconds": round(duration, 3),
        "chunk_seconds": chunk_seconds,
        "requested_overlap_seconds": chunk_overlap_seconds,
        "effective_overlap_seconds": effective_overlap,
        "strategy": strategy,
        "windows": [asdict(item) for item in windows],
        "speech_segments": speech_segments,
        "silence_gaps": silence_gaps,
        "silence_cuts": cut_decisions,
        "warnings": warnings,
        "vad_output_excerpt": "\n".join(vad_output.splitlines()[:80]),
    }


def _run_case_transcription(
    *,
    pair: GoldenPair,
    job_dir: Path,
    settings: Settings,
    model_path: Path,
    language: str,
    beam_size: int,
    best_of: int,
    vad_threshold: float,
    vad_cut_threshold: float,
    vad_max_speech_duration_s: int,
    vad_min_silence_duration_ms: int,
    vad_speech_pad_ms: int,
    chunk_seconds: int,
    chunk_overlap_seconds: int,
    stitch_method: str | None,
    stitch_methods: list[str] | None,
) -> dict[str, Any]:
    job_dir.mkdir(parents=True, exist_ok=True)
    return run_transcription(
        job_id=_safe_slug(f"{pair.name}-{chunk_seconds}-{time.time_ns()}")[:32],
        job_dir=job_dir,
        input_path=pair.audio_path,
        settings=settings,
        model_path=model_path,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        vad_threshold=vad_threshold,
        vad_cut_threshold=vad_cut_threshold,
        vad_max_speech_duration_s=vad_max_speech_duration_s,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        chunk_seconds=chunk_seconds,
        chunk_overlap_seconds=chunk_overlap_seconds,
        stitch_method=stitch_method,
        stitch_methods=stitch_methods,
        repetition_guard=settings.repetition_guard,
        set_progress=lambda _progress: None,
    )


def _write_text_evaluation(
    *,
    case_dir: Path,
    reference_text: str,
    hypothesis_text: str,
    result: dict[str, Any],
    baseline_text: str | None = None,
) -> WerResult:
    case_dir.mkdir(parents=True, exist_ok=True)
    wer = compute_wer(reference_text, hypothesis_text)
    payload = {
        "wer": {
            "wer": round(wer.wer, 6),
            "substitutions": wer.substitutions,
            "deletions": wer.deletions,
            "insertions": wer.insertions,
            "reference_words": wer.reference_words,
            "hypothesis_words": wer.hypothesis_words,
        },
        "reference_text": reference_text,
        "hypothesis_text": hypothesis_text,
        "reference_normalized": normalize_for_wer(reference_text),
        "hypothesis_normalized": normalize_for_wer(hypothesis_text),
        "operations": wer.operations,
        "result_decode": result.get("decode", {}),
        "warnings": result.get("warnings", []),
    }
    if baseline_text is not None and baseline_text != hypothesis_text:
        payload["hypothesis_vs_full_transcript"] = {
            "wer": round(compute_wer(baseline_text, hypothesis_text).wer, 6)
        }
    _write_json(case_dir / "evaluation.json", payload)
    return wer


def _summary_row(
    *,
    pair: GoldenPair,
    mode: str,
    stitch_method: str,
    duration: float,
    chunk_seconds: int,
    requested_overlap_seconds: int,
    result: dict[str, Any],
    wer: WerResult,
    full_wer: WerResult | None,
    debug_path: Path,
) -> dict[str, Any]:
    chunking = (result.get("decode") or {}).get("chunking") or {}
    warnings = result.get("warnings") or []
    fallback = next((item for item in warnings if item.get("type") == "chunking_fixed_fallback"), None)
    return {
        "case": pair.name,
        "mode": mode,
        "stitch_method": stitch_method,
        "duration_seconds": round(duration, 3),
        "chunk_seconds": chunk_seconds,
        "requested_overlap_seconds": requested_overlap_seconds,
        "effective_overlap_seconds": chunking.get("overlap_seconds", 0),
        "strategy": chunking.get("strategy", "single" if chunk_seconds == 0 else ""),
        "chunk_count": chunking.get("chunk_count", 1 if chunk_seconds == 0 else ""),
        "fallback_reason": "" if fallback is None else fallback.get("reason", ""),
        "wer": round(wer.wer, 6),
        "wer_delta_vs_full": "" if full_wer is None else round(wer.wer - full_wer.wer, 6),
        "substitutions": wer.substitutions,
        "deletions": wer.deletions,
        "insertions": wer.insertions,
        "reference_words": wer.reference_words,
        "hypothesis_words": wer.hypothesis_words,
        "elapsed_seconds": (result.get("metrics") or {}).get("elapsed_seconds", ""),
        "rtf": (result.get("metrics") or {}).get("rtf", ""),
        "result_path": str(debug_path.parent / "result.json") if (debug_path.parent / "result.json").exists() else "",
        "debug_path": str(debug_path),
    }


def _planner_summary_row(
    *,
    pair: GoldenPair,
    duration: float,
    chunk_seconds: int,
    requested_overlap_seconds: int,
    planner: dict[str, Any],
    debug_path: Path,
) -> dict[str, Any]:
    fallback = next((item for item in planner.get("warnings", []) if item.get("type") == "chunking_fixed_fallback"), None)
    return {
        "case": pair.name,
        "mode": f"planner_chunk_{chunk_seconds}",
        "stitch_method": "",
        "duration_seconds": round(duration, 3),
        "chunk_seconds": chunk_seconds,
        "requested_overlap_seconds": requested_overlap_seconds,
        "effective_overlap_seconds": planner.get("effective_overlap_seconds", ""),
        "strategy": planner.get("strategy", ""),
        "chunk_count": len(planner.get("windows") or []),
        "fallback_reason": "" if fallback is None else fallback.get("reason", ""),
        "wer": "",
        "wer_delta_vs_full": "",
        "substitutions": "",
        "deletions": "",
        "insertions": "",
        "reference_words": "",
        "hypothesis_words": "",
        "elapsed_seconds": "",
        "rtf": "",
        "result_path": "",
        "debug_path": str(debug_path),
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Golden Cut Debug Summary", ""]
    if not rows:
        lines.append("No golden cases were found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    chunk_rows = [row for row in rows if row.get("wer_delta_vs_full") not in {"", None}]
    chunk_rows.sort(key=lambda item: float(item.get("wer_delta_vs_full") or 0), reverse=True)
    fallback_rows = [row for row in rows if row.get("fallback_reason")]
    lines.extend(
        [
            f"- Rows: {len(rows)}",
            f"- Fixed fallback rows: {len(fallback_rows)}",
            "",
            "## Largest WER Regressions",
            "",
        ]
    )
    if chunk_rows:
        for row in chunk_rows[:20]:
            lines.append(
                (
                    f"- `{row['case']}` {row['mode']} {row['stitch_method']}: "
                    f"WER {row['wer']} (delta {row['wer_delta_vs_full']}), "
                    f"strategy `{row['strategy']}`, debug `{row['debug_path']}`"
                )
            )
    else:
        lines.append("- No WER rows available. Planner-only run.")

    if fallback_rows:
        lines.extend(["", "## Fixed Fallbacks", ""])
        for row in fallback_rows[:30]:
            lines.append(
                f"- `{row['case']}` {row['mode']}: {row['fallback_reason']} (`{row['debug_path']}`)"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "case"
