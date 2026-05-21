import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable


@dataclass(frozen=True)
class TokenOverlap:
    token_count: int
    covered_count: int
    prefix_count: int
    suffix_count: int
    longest_run: int

    @property
    def coverage(self) -> float:
        if self.token_count <= 0:
            return 0.0
        return self.covered_count / self.token_count


@dataclass(frozen=True)
class WordEntry:
    token: str
    word: str
    start: float
    end: float
    segment_index: int
    word_index: int


@dataclass(frozen=True)
class WordAlignment:
    trim_word_count: int
    existing_start: int
    existing_end: int
    incoming_start: int
    incoming_end: int
    match_token_count: int
    score: float


@dataclass(frozen=True)
class CenterSeam:
    seam_time: float
    incoming_trim_word_count: int
    existing_start: int
    existing_end: int
    incoming_start: int
    incoming_end: int
    match_offset: int
    match_token_count: int
    score: float


MIN_OVERLAP_TOKENS = 4
DROP_OVERLAP_COVERAGE = 0.86
BOUNDARY_CONTEXT_SECONDS = 60.0
BOUNDARY_CONTEXT_SEGMENTS = 40
WORD_ALIGN_MIN_MATCH_TOKENS = 4
WORD_ALIGN_INCOMING_CONTEXT_SECONDS = 90.0
STITCH_METHODS = {"fuzzy", "safe_zone", "word_align", "center_align"}

StitchStrategy = Callable[..., tuple[list[dict[str, Any]], dict[str, Any] | None]]


def normalize_stitch_method(value: str | None, default: str = "fuzzy") -> str:
    method = (value or default).strip().lower()
    if method in STITCH_METHODS:
        return method
    return default


def merge_chunk_segments(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    overlap_start_seconds: float,
    overlap_end_seconds: float | None,
    stitch_method: str = "fuzzy",
) -> list[dict[str, Any]]:
    merged, _audit = merge_chunk_segments_with_audit(
        existing,
        incoming,
        previous_chunk_index=None,
        next_chunk_index=None,
        overlap_start_seconds=overlap_start_seconds,
        overlap_end_seconds=overlap_end_seconds,
        overlap_seconds=0,
        chunk_start_seconds=overlap_start_seconds,
        chunk_end_seconds=overlap_end_seconds,
        is_first_chunk=overlap_end_seconds is None,
        is_last_chunk=False,
        incoming_warning=None,
        stitch_method=stitch_method,
    )
    return merged


def merge_chunk_segments_with_audit(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    previous_chunk_index: int | None,
    next_chunk_index: int | None,
    overlap_start_seconds: float,
    overlap_end_seconds: float | None,
    overlap_seconds: float = 0,
    leading_overlap_seconds: float | None = None,
    trailing_overlap_seconds: float | None = None,
    chunk_start_seconds: float | None = None,
    chunk_end_seconds: float | None = None,
    is_first_chunk: bool = False,
    is_last_chunk: bool = False,
    incoming_warning: dict[str, Any] | None,
    stitch_method: str = "fuzzy",
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    method = normalize_stitch_method(stitch_method)
    if method == "safe_zone":
        return merge_chunk_segments_safe_zone(
            existing,
            incoming,
            previous_chunk_index=previous_chunk_index,
            next_chunk_index=next_chunk_index,
            overlap_start_seconds=overlap_start_seconds,
            overlap_end_seconds=overlap_end_seconds,
            overlap_seconds=overlap_seconds,
            leading_overlap_seconds=leading_overlap_seconds,
            trailing_overlap_seconds=trailing_overlap_seconds,
            chunk_start_seconds=chunk_start_seconds,
            chunk_end_seconds=chunk_end_seconds,
            is_first_chunk=is_first_chunk,
            is_last_chunk=is_last_chunk,
            incoming_warning=incoming_warning,
        )
    if method == "word_align":
        return merge_chunk_segments_word_align(
            existing,
            incoming,
            previous_chunk_index=previous_chunk_index,
            next_chunk_index=next_chunk_index,
            overlap_start_seconds=overlap_start_seconds,
            overlap_end_seconds=overlap_end_seconds,
            incoming_warning=incoming_warning,
        )
    if method == "center_align":
        return merge_chunk_segments_center_align(
            existing,
            incoming,
            previous_chunk_index=previous_chunk_index,
            next_chunk_index=next_chunk_index,
            overlap_start_seconds=overlap_start_seconds,
            overlap_end_seconds=overlap_end_seconds,
            incoming_warning=incoming_warning,
        )
    return merge_chunk_segments_fuzzy(
        existing,
        incoming,
        previous_chunk_index=previous_chunk_index,
        next_chunk_index=next_chunk_index,
        overlap_start_seconds=overlap_start_seconds,
        overlap_end_seconds=overlap_end_seconds,
        incoming_warning=incoming_warning,
    )


def merge_chunk_segments_fuzzy(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    previous_chunk_index: int | None,
    next_chunk_index: int | None,
    overlap_start_seconds: float,
    overlap_end_seconds: float | None,
    incoming_warning: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not existing or overlap_end_seconds is None:
        return existing + incoming, None

    merged = list(existing)
    incoming_decisions: list[dict[str, Any]] = []
    dropped_segments: list[dict[str, Any]] = []
    trimmed_segments: list[dict[str, Any]] = []
    kept_segments: list[dict[str, Any]] = []
    kept_overlap_segments: list[dict[str, Any]] = []
    kept_outside_overlap_segments: list[dict[str, Any]] = []
    dropped_exact_count = 0
    dropped_overlap_count = 0
    trimmed_prefix_count = 0
    trimmed_suffix_count = 0
    kept_low_confidence_count = 0
    for segment in incoming:
        segment_text = _normalize_for_compare(segment["transcript"])
        overlap_relevant = segment["start"] < overlap_end_seconds and segment["end"] > overlap_start_seconds
        context_segments = _boundary_context_segments(merged, overlap_start_seconds)
        recent_texts = {
            _normalize_for_compare(item["transcript"])
            for item in context_segments
            if _normalize_for_compare(item["transcript"])
        }

        if overlap_relevant and segment_text and segment_text in recent_texts:
            dropped_segments.append(_audit_segment(segment))
            dropped_exact_count += 1
            incoming_decisions.append(
                {
                    "decision": "dropped_exact_duplicate",
                    "segment": _audit_segment(segment),
                }
            )
            continue

        if overlap_relevant:
            context_tokens = _comparison_context_tokens(context_segments)
            segment_tokens = _tokenize_for_compare(segment["transcript"])
            token_overlap = _measure_token_overlap(context_tokens, segment_tokens)
            if _should_drop_overlap_duplicate(token_overlap):
                dropped_segments.append(_audit_segment(segment))
                dropped_overlap_count += 1
                incoming_decisions.append(
                    {
                        "decision": "dropped_overlap_duplicate",
                        "segment": _audit_segment(segment),
                        "overlap": _audit_token_overlap(token_overlap),
                    }
                )
                continue

            trimmed_segment: dict[str, Any] | None = None
            decision = "kept_low_confidence_overlap"
            if _should_trim_duplicate_prefix(token_overlap):
                trimmed_segment = _trim_segment_tokens(segment, trim_prefix_tokens=token_overlap.prefix_count)
                decision = "trimmed_duplicate_prefix"
                trimmed_prefix_count += 1
            elif _should_trim_duplicate_suffix(token_overlap):
                trimmed_segment = _trim_segment_tokens(segment, trim_suffix_tokens=token_overlap.suffix_count)
                decision = "trimmed_duplicate_suffix"
                trimmed_suffix_count += 1

            if trimmed_segment is not None:
                audited = _audit_segment(trimmed_segment)
                trimmed_segments.append(audited)
                kept_segments.append(audited)
                kept_overlap_segments.append(audited)
                incoming_decisions.append(
                    {
                        "decision": decision,
                        "segment": audited,
                        "original_segment": _audit_segment(segment),
                        "overlap": _audit_token_overlap(token_overlap),
                    }
                )
                merged.append(trimmed_segment)
                continue

            kept_low_confidence_count += 1
        else:
            decision = "kept_outside_overlap"

        audited = _audit_segment(segment)
        kept_segments.append(audited)
        if overlap_relevant:
            kept_overlap_segments.append(audited)
        else:
            kept_outside_overlap_segments.append(audited)
        incoming_decisions.append(
            {
                "decision": decision,
                "segment": audited,
            }
        )
        merged.append(segment)

    audit = _base_audit(
        method="fuzzy",
        previous_chunk_index=previous_chunk_index,
        next_chunk_index=next_chunk_index,
        overlap_start_seconds=overlap_start_seconds,
        overlap_end_seconds=overlap_end_seconds,
        existing=existing,
        incoming_decisions=incoming_decisions,
        dropped_segments=dropped_segments,
        trimmed_segments=trimmed_segments,
        kept_overlap_segments=kept_overlap_segments,
        kept_outside_overlap_segments=kept_outside_overlap_segments,
        incoming_warning=incoming_warning,
        counts={
            "incoming": len(incoming),
            "dropped_duplicates": len(dropped_segments),
            "dropped_exact_duplicates": dropped_exact_count,
            "dropped_overlap_duplicates": dropped_overlap_count,
            "dropped_safe_zone": 0,
            "trimmed_duplicate_prefix": trimmed_prefix_count,
            "trimmed_duplicate_suffix": trimmed_suffix_count,
            "kept": len(kept_segments),
            "kept_overlap": len(kept_overlap_segments),
            "kept_outside_overlap": len(kept_outside_overlap_segments),
            "kept_low_confidence_overlap": kept_low_confidence_count,
        },
    )
    return merged, audit


def merge_chunk_segments_word_align(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    previous_chunk_index: int | None,
    next_chunk_index: int | None,
    overlap_start_seconds: float,
    overlap_end_seconds: float | None,
    incoming_warning: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not existing or overlap_end_seconds is None:
        return existing + incoming, None

    context_segments = _boundary_context_segments(existing, overlap_start_seconds)
    incoming_context_segments = [
        item for item in incoming if float(item["start"]) <= overlap_end_seconds + WORD_ALIGN_INCOMING_CONTEXT_SECONDS
    ]
    existing_words = _collect_word_entries(context_segments)
    incoming_words = _collect_word_entries(incoming_context_segments)
    alignment = _find_word_alignment(
        existing_words=existing_words,
        incoming_words=incoming_words,
        overlap_start_seconds=overlap_start_seconds,
        overlap_end_seconds=overlap_end_seconds,
    )

    if alignment is None:
        merged, audit = merge_chunk_segments_fuzzy(
            existing,
            incoming,
            previous_chunk_index=previous_chunk_index,
            next_chunk_index=next_chunk_index,
            overlap_start_seconds=overlap_start_seconds,
            overlap_end_seconds=overlap_end_seconds,
            incoming_warning=incoming_warning,
        )
        if audit is not None:
            audit["method"] = "word_align"
            audit["fallback_method"] = "fuzzy"
            audit["word_alignment"] = None
        return merged, audit

    word_trimmed_incoming, word_dropped_segments, word_trimmed_segments = _trim_incoming_prefix_by_word_count(
        incoming,
        trim_word_count=alignment.trim_word_count,
    )

    merged, audit = merge_chunk_segments_fuzzy(
        existing,
        word_trimmed_incoming,
        previous_chunk_index=previous_chunk_index,
        next_chunk_index=next_chunk_index,
        overlap_start_seconds=overlap_start_seconds,
        overlap_end_seconds=overlap_end_seconds,
        incoming_warning=incoming_warning,
    )
    if audit is None:
        return merged, None
    audit["method"] = "word_align"
    audit["word_alignment"] = _audit_word_alignment(alignment, existing_words, incoming_words)
    audit["word_aligned_prefix"] = {
        "incoming_before_word_align": len(incoming),
        "incoming_after_word_align": len(word_trimmed_incoming),
        "dropped_segments": word_dropped_segments,
        "trimmed_segments": word_trimmed_segments,
    }
    audit["counts"]["incoming_before_word_align"] = len(incoming)
    audit["counts"]["word_aligned_prefix_dropped"] = len(word_dropped_segments)
    audit["counts"]["word_aligned_prefix_trimmed"] = len(word_trimmed_segments)
    return merged, audit


def merge_chunk_segments_center_align(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    previous_chunk_index: int | None,
    next_chunk_index: int | None,
    overlap_start_seconds: float,
    overlap_end_seconds: float | None,
    incoming_warning: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not existing or overlap_end_seconds is None:
        return existing + incoming, None

    context_segments = _boundary_context_segments(existing, overlap_start_seconds)
    incoming_context_segments = [
        item for item in incoming if float(item["start"]) <= overlap_end_seconds + WORD_ALIGN_INCOMING_CONTEXT_SECONDS
    ]
    existing_words = _collect_word_entries(context_segments)
    incoming_words = _collect_word_entries(incoming_context_segments)
    seam = _find_center_seam(
        existing_words=existing_words,
        incoming_words=incoming_words,
        overlap_start_seconds=overlap_start_seconds,
        overlap_end_seconds=overlap_end_seconds,
    )

    if seam is None:
        merged, audit = merge_chunk_segments_fuzzy(
            existing,
            incoming,
            previous_chunk_index=previous_chunk_index,
            next_chunk_index=next_chunk_index,
            overlap_start_seconds=overlap_start_seconds,
            overlap_end_seconds=overlap_end_seconds,
            incoming_warning=incoming_warning,
        )
        if audit is not None:
            audit["method"] = "center_align"
            audit["fallback_method"] = "fuzzy"
            audit["center_alignment"] = None
        return merged, audit

    trimmed_existing, existing_dropped, existing_trimmed = _trim_existing_suffix_at_time(
        existing,
        seam_time=seam.seam_time,
    )
    trimmed_incoming, incoming_dropped, incoming_trimmed = _trim_incoming_prefix_by_word_count(
        incoming,
        trim_word_count=seam.incoming_trim_word_count,
    )
    merged, audit = merge_chunk_segments_fuzzy(
        trimmed_existing,
        trimmed_incoming,
        previous_chunk_index=previous_chunk_index,
        next_chunk_index=next_chunk_index,
        overlap_start_seconds=overlap_start_seconds,
        overlap_end_seconds=overlap_end_seconds,
        incoming_warning=incoming_warning,
    )
    if audit is None:
        return merged, None
    audit["method"] = "center_align"
    audit["center_alignment"] = _audit_center_seam(seam, existing_words, incoming_words)
    audit["center_trim"] = {
        "existing_before_center_align": len(existing),
        "existing_after_center_align": len(trimmed_existing),
        "incoming_before_center_align": len(incoming),
        "incoming_after_center_align": len(trimmed_incoming),
        "existing_dropped_segments": existing_dropped,
        "existing_trimmed_segments": existing_trimmed,
        "incoming_dropped_segments": incoming_dropped,
        "incoming_trimmed_segments": incoming_trimmed,
    }
    audit["counts"]["existing_center_dropped"] = len(existing_dropped)
    audit["counts"]["existing_center_trimmed"] = len(existing_trimmed)
    audit["counts"]["incoming_center_dropped"] = len(incoming_dropped)
    audit["counts"]["incoming_center_trimmed"] = len(incoming_trimmed)
    return merged, audit


def merge_chunk_segments_safe_zone(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    previous_chunk_index: int | None,
    next_chunk_index: int | None,
    overlap_start_seconds: float,
    overlap_end_seconds: float | None,
    overlap_seconds: float,
    leading_overlap_seconds: float | None,
    trailing_overlap_seconds: float | None,
    chunk_start_seconds: float | None,
    chunk_end_seconds: float | None,
    is_first_chunk: bool,
    is_last_chunk: bool,
    incoming_warning: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    chunk_start = overlap_start_seconds if chunk_start_seconds is None else chunk_start_seconds
    fallback_end = max((float(item["end"]) for item in incoming), default=chunk_start)
    chunk_end = fallback_end if chunk_end_seconds is None else chunk_end_seconds
    leading_half_overlap = max(float(overlap_seconds if leading_overlap_seconds is None else leading_overlap_seconds), 0.0) / 2.0
    trailing_half_overlap = max(float(overlap_seconds if trailing_overlap_seconds is None else trailing_overlap_seconds), 0.0) / 2.0
    safe_start = chunk_start if is_first_chunk else chunk_start + leading_half_overlap
    safe_end = chunk_end if is_last_chunk else chunk_end - trailing_half_overlap

    merged = list(existing)
    incoming_decisions: list[dict[str, Any]] = []
    dropped_segments: list[dict[str, Any]] = []
    kept_segments: list[dict[str, Any]] = []
    kept_overlap_segments: list[dict[str, Any]] = []
    kept_outside_overlap_segments: list[dict[str, Any]] = []
    for segment in incoming:
        start = float(segment["start"])
        end = float(segment["end"])
        midpoint = (start + end) / 2.0
        in_safe_zone = safe_start <= midpoint and (midpoint <= safe_end if is_last_chunk else midpoint < safe_end)
        overlap_relevant = (
            overlap_end_seconds is not None
            and segment["start"] < overlap_end_seconds
            and segment["end"] > overlap_start_seconds
        )
        audited = _audit_segment(segment)
        audited["midpoint"] = round(midpoint, 3)
        audited["midpoint_label"] = format_timestamp(midpoint)

        if in_safe_zone:
            kept_segments.append(audited)
            if overlap_relevant:
                kept_overlap_segments.append(audited)
            else:
                kept_outside_overlap_segments.append(audited)
            incoming_decisions.append({"decision": "kept_safe_zone", "segment": audited})
            merged.append(segment)
            continue

        dropped_segments.append(audited)
        incoming_decisions.append({"decision": "dropped_safe_zone", "segment": audited})

    if not existing or overlap_end_seconds is None:
        return merged, None

    audit = _base_audit(
        method="safe_zone",
        previous_chunk_index=previous_chunk_index,
        next_chunk_index=next_chunk_index,
        overlap_start_seconds=overlap_start_seconds,
        overlap_end_seconds=overlap_end_seconds,
        existing=existing,
        incoming_decisions=incoming_decisions,
        dropped_segments=dropped_segments,
        trimmed_segments=[],
        kept_overlap_segments=kept_overlap_segments,
        kept_outside_overlap_segments=kept_outside_overlap_segments,
        incoming_warning=incoming_warning,
        counts={
            "incoming": len(incoming),
            "dropped_duplicates": 0,
            "dropped_exact_duplicates": 0,
            "dropped_overlap_duplicates": 0,
            "dropped_safe_zone": len(dropped_segments),
            "trimmed_duplicate_prefix": 0,
            "trimmed_duplicate_suffix": 0,
            "kept": len(kept_segments),
            "kept_overlap": len(kept_overlap_segments),
            "kept_outside_overlap": len(kept_outside_overlap_segments),
            "kept_low_confidence_overlap": 0,
        },
    )
    audit["safe_start"] = round(safe_start, 3)
    audit["safe_end"] = round(safe_end, 3)
    audit["safe_start_label"] = format_timestamp(safe_start)
    audit["safe_end_label"] = format_timestamp(safe_end)
    return merged, audit


def _base_audit(
    *,
    method: str,
    previous_chunk_index: int | None,
    next_chunk_index: int | None,
    overlap_start_seconds: float,
    overlap_end_seconds: float,
    existing: list[dict[str, Any]],
    incoming_decisions: list[dict[str, Any]],
    dropped_segments: list[dict[str, Any]],
    trimmed_segments: list[dict[str, Any]],
    kept_overlap_segments: list[dict[str, Any]],
    kept_outside_overlap_segments: list[dict[str, Any]],
    incoming_warning: dict[str, Any] | None,
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "method": method,
        "previous_chunk": previous_chunk_index,
        "next_chunk": next_chunk_index,
        "overlap_start": round(overlap_start_seconds, 3),
        "overlap_end": round(overlap_end_seconds, 3),
        "overlap_start_label": format_timestamp(overlap_start_seconds),
        "overlap_end_label": format_timestamp(overlap_end_seconds),
        "previous_tail": [_audit_segment(item) for item in existing[-8:]],
        "incoming_head": incoming_decisions[:8],
        "dropped_duplicates": dropped_segments,
        "trimmed_duplicates": trimmed_segments,
        "kept_overlap": kept_overlap_segments,
        "kept_outside_overlap_head": kept_outside_overlap_segments[:8],
        "incoming_warning": incoming_warning,
        "counts": counts,
    }


def _boundary_context_segments(
    segments: list[dict[str, Any]],
    overlap_start_seconds: float,
) -> list[dict[str, Any]]:
    context_start = overlap_start_seconds - BOUNDARY_CONTEXT_SECONDS
    nearby_segments = [item for item in segments if item["end"] >= context_start]
    return nearby_segments[-BOUNDARY_CONTEXT_SEGMENTS:]


def _comparison_context_tokens(segments: list[dict[str, Any]]) -> list[str]:
    return _tokenize_for_compare(" ".join(item.get("transcript", "") for item in segments))


def _tokenize_for_compare(text: str) -> list[str]:
    normalized = _normalize_for_compare(text)
    if not normalized:
        return []
    return normalized.split()


def _measure_token_overlap(context_tokens: list[str], segment_tokens: list[str]) -> TokenOverlap:
    if not context_tokens or not segment_tokens:
        return TokenOverlap(
            token_count=len(segment_tokens),
            covered_count=0,
            prefix_count=0,
            suffix_count=0,
            longest_run=0,
        )

    covered = [False] * len(segment_tokens)
    longest_run = 0
    matcher = SequenceMatcher(None, context_tokens, segment_tokens, autojunk=False)
    for _context_start, segment_start, size in matcher.get_matching_blocks():
        if size <= 0:
            continue
        longest_run = max(longest_run, size)
        for index in range(segment_start, segment_start + size):
            covered[index] = True

    prefix_count = 0
    for is_covered in covered:
        if not is_covered:
            break
        prefix_count += 1

    suffix_count = 0
    for is_covered in reversed(covered):
        if not is_covered:
            break
        suffix_count += 1

    return TokenOverlap(
        token_count=len(segment_tokens),
        covered_count=sum(1 for is_covered in covered if is_covered),
        prefix_count=prefix_count,
        suffix_count=suffix_count,
        longest_run=longest_run,
    )


def _collect_word_entries(segments: list[dict[str, Any]]) -> list[WordEntry]:
    entries: list[WordEntry] = []
    for segment_index, segment in enumerate(segments):
        words = segment.get("words") or []
        if not isinstance(words, list):
            continue
        for word_index, word in enumerate(words):
            if not isinstance(word, dict):
                continue
            token = _normalize_for_compare(str(word.get("word", "")))
            if not token:
                continue
            try:
                start = float(word.get("start", segment.get("start", 0.0)))
                end = float(word.get("end", word.get("start", segment.get("end", 0.0))))
            except (TypeError, ValueError):
                continue
            entries.append(
                WordEntry(
                    token=token,
                    word=str(word.get("word", "")),
                    start=round(start, 3),
                    end=round(max(start, end), 3),
                    segment_index=segment_index,
                    word_index=word_index,
                )
            )
    return entries


def _find_word_alignment(
    *,
    existing_words: list[WordEntry],
    incoming_words: list[WordEntry],
    overlap_start_seconds: float,
    overlap_end_seconds: float,
) -> WordAlignment | None:
    if not existing_words or not incoming_words:
        return None

    existing_tokens = [item.token for item in existing_words]
    incoming_tokens = [item.token for item in incoming_words]
    matcher = SequenceMatcher(None, existing_tokens, incoming_tokens, autojunk=False)
    candidates: list[WordAlignment] = []
    for existing_start, incoming_start, size in matcher.get_matching_blocks():
        if size < WORD_ALIGN_MIN_MATCH_TOKENS:
            continue
        incoming_end = incoming_start + size
        existing_end = existing_start + size
        incoming_match_start = incoming_words[incoming_start].start
        incoming_match_end = incoming_words[incoming_end - 1].end
        existing_match_end = existing_words[existing_end - 1].end
        if incoming_match_start > overlap_end_seconds:
            continue
        if incoming_match_end < overlap_start_seconds:
            continue
        if existing_match_end < overlap_start_seconds:
            continue

        incoming_overlap_bonus = max(overlap_end_seconds - incoming_match_start, 0.0)
        incoming_prefix_penalty = incoming_start * 0.25
        existing_tail_bonus = existing_end / max(len(existing_words), 1)
        score = size + min(incoming_overlap_bonus, 5.0) + existing_tail_bonus - incoming_prefix_penalty
        candidates.append(
            WordAlignment(
                trim_word_count=incoming_end,
                existing_start=existing_start,
                existing_end=existing_end,
                incoming_start=incoming_start,
                incoming_end=incoming_end,
                match_token_count=size,
                score=score,
            )
        )

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.score,
            item.match_token_count,
            item.incoming_end,
            item.existing_end,
            -item.incoming_start,
        ),
    )


def _find_center_seam(
    *,
    existing_words: list[WordEntry],
    incoming_words: list[WordEntry],
    overlap_start_seconds: float,
    overlap_end_seconds: float,
) -> CenterSeam | None:
    if not existing_words or not incoming_words:
        return None

    overlap_midpoint = (overlap_start_seconds + overlap_end_seconds) / 2.0
    existing_tokens = [item.token for item in existing_words]
    incoming_tokens = [item.token for item in incoming_words]
    matcher = SequenceMatcher(None, existing_tokens, incoming_tokens, autojunk=False)
    candidates: list[CenterSeam] = []
    for existing_start, incoming_start, size in matcher.get_matching_blocks():
        if size < WORD_ALIGN_MIN_MATCH_TOKENS:
            continue
        for offset in range(size):
            existing_word = existing_words[existing_start + offset]
            incoming_word = incoming_words[incoming_start + offset]
            seam_time = (existing_word.end + incoming_word.end) / 2.0
            if seam_time < overlap_start_seconds or seam_time > overlap_end_seconds:
                continue

            distance = abs(seam_time - overlap_midpoint)
            edge_distance = min(seam_time - overlap_start_seconds, overlap_end_seconds - seam_time)
            score = min(size, 40) + min(edge_distance, 5.0) - distance
            candidates.append(
                CenterSeam(
                    seam_time=round(seam_time, 3),
                    incoming_trim_word_count=incoming_start + offset + 1,
                    existing_start=existing_start,
                    existing_end=existing_start + size,
                    incoming_start=incoming_start,
                    incoming_end=incoming_start + size,
                    match_offset=offset,
                    match_token_count=size,
                    score=score,
                )
            )

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.score,
            -abs(item.seam_time - overlap_midpoint),
            item.match_token_count,
            item.seam_time,
        ),
    )


def _trim_incoming_prefix_by_word_count(
    incoming: list[dict[str, Any]],
    *,
    trim_word_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped_segments: list[dict[str, Any]] = []
    trimmed_segments: list[dict[str, Any]] = []
    remaining_to_trim = max(trim_word_count, 0)

    for segment in incoming:
        words = segment.get("words") or []
        if not isinstance(words, list) or not words:
            if remaining_to_trim > 0:
                dropped_segments.append(_audit_segment(segment))
                continue
            kept.append(segment)
            continue

        word_count = len(words)
        if remaining_to_trim >= word_count:
            dropped_segments.append(_audit_segment(segment))
            remaining_to_trim -= word_count
            continue
        if remaining_to_trim <= 0:
            kept.append(segment)
            continue

        trimmed = _trim_segment_by_word_range(
            segment,
            keep_start_word=remaining_to_trim,
            keep_end_word=word_count,
        )
        remaining_to_trim = 0
        if trimmed is None:
            dropped_segments.append(_audit_segment(segment))
            continue
        audited = _audit_segment(trimmed)
        trimmed_segments.append(audited)
        kept.append(trimmed)

    return kept, dropped_segments, trimmed_segments


def _trim_existing_suffix_at_time(
    existing: list[dict[str, Any]],
    *,
    seam_time: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped_segments: list[dict[str, Any]] = []
    trimmed_segments: list[dict[str, Any]] = []

    for segment in existing:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if end <= seam_time:
            kept.append(segment)
            continue
        if start >= seam_time:
            dropped_segments.append(_audit_segment(segment))
            continue

        words = segment.get("words") or []
        if not isinstance(words, list) or not words:
            midpoint = (start + end) / 2.0
            if midpoint <= seam_time:
                kept.append(segment)
            else:
                dropped_segments.append(_audit_segment(segment))
            continue

        keep_end_word = 0
        for index, word in enumerate(words, start=1):
            try:
                word_end = float(word.get("end", word.get("start", end)))
            except (TypeError, ValueError):
                continue
            if word_end <= seam_time:
                keep_end_word = index

        if keep_end_word <= 0:
            dropped_segments.append(_audit_segment(segment))
            continue
        if keep_end_word >= len(words):
            kept.append(segment)
            continue

        trimmed = _trim_segment_by_word_range(segment, keep_start_word=0, keep_end_word=keep_end_word)
        if trimmed is None:
            dropped_segments.append(_audit_segment(segment))
            continue
        audited = _audit_segment(trimmed)
        trimmed_segments.append(audited)
        kept.append(trimmed)

    return kept, dropped_segments, trimmed_segments


def _trim_segment_by_word_range(
    segment: dict[str, Any],
    *,
    keep_start_word: int,
    keep_end_word: int,
) -> dict[str, Any] | None:
    words = segment.get("words") or []
    keep_start_word = min(max(keep_start_word, 0), len(words))
    keep_end_word = max(min(keep_end_word, len(words)), 0)
    if keep_start_word >= keep_end_word:
        return None

    kept_words = [_clip_word(word, start=None, end=None) for word in words[keep_start_word:keep_end_word]]
    kept_words = [word for word in kept_words if word is not None]
    if not kept_words:
        return None
    transcript = " ".join(str(word["word"]) for word in kept_words).strip()
    if not transcript:
        return None

    start = min(float(word["start"]) for word in kept_words)
    end = max(float(word["end"]) for word in kept_words)
    trimmed = dict(segment)
    trimmed["start"] = round(start, 3)
    trimmed["end"] = round(max(start, end), 3)
    trimmed["transcript"] = transcript
    trimmed["words"] = kept_words
    return trimmed


def _audit_word_alignment(
    alignment: WordAlignment,
    existing_words: list[WordEntry],
    incoming_words: list[WordEntry],
) -> dict[str, Any]:
    existing_match = existing_words[alignment.existing_start : alignment.existing_end]
    incoming_match = incoming_words[alignment.incoming_start : alignment.incoming_end]
    return {
        "trim_word_count": alignment.trim_word_count,
        "match_token_count": alignment.match_token_count,
        "score": round(alignment.score, 3),
        "existing_start": alignment.existing_start,
        "existing_end": alignment.existing_end,
        "incoming_start": alignment.incoming_start,
        "incoming_end": alignment.incoming_end,
        "existing_text": " ".join(item.word for item in existing_match),
        "incoming_text": " ".join(item.word for item in incoming_match),
        "existing_start_time": None if not existing_match else round(existing_match[0].start, 3),
        "existing_end_time": None if not existing_match else round(existing_match[-1].end, 3),
        "incoming_start_time": None if not incoming_match else round(incoming_match[0].start, 3),
        "incoming_end_time": None if not incoming_match else round(incoming_match[-1].end, 3),
    }


def _audit_center_seam(
    seam: CenterSeam,
    existing_words: list[WordEntry],
    incoming_words: list[WordEntry],
) -> dict[str, Any]:
    existing_match = existing_words[seam.existing_start : seam.existing_end]
    incoming_match = incoming_words[seam.incoming_start : seam.incoming_end]
    return {
        "seam_time": seam.seam_time,
        "seam_label": format_timestamp(seam.seam_time),
        "incoming_trim_word_count": seam.incoming_trim_word_count,
        "match_token_count": seam.match_token_count,
        "match_offset": seam.match_offset,
        "score": round(seam.score, 3),
        "existing_start": seam.existing_start,
        "existing_end": seam.existing_end,
        "incoming_start": seam.incoming_start,
        "incoming_end": seam.incoming_end,
        "existing_text": " ".join(item.word for item in existing_match),
        "incoming_text": " ".join(item.word for item in incoming_match),
        "existing_start_time": None if not existing_match else round(existing_match[0].start, 3),
        "existing_end_time": None if not existing_match else round(existing_match[-1].end, 3),
        "incoming_start_time": None if not incoming_match else round(incoming_match[0].start, 3),
        "incoming_end_time": None if not incoming_match else round(incoming_match[-1].end, 3),
    }


def _should_drop_overlap_duplicate(token_overlap: TokenOverlap) -> bool:
    if token_overlap.token_count < MIN_OVERLAP_TOKENS:
        return False
    return (
        token_overlap.coverage >= DROP_OVERLAP_COVERAGE
        and token_overlap.longest_run >= min(MIN_OVERLAP_TOKENS, token_overlap.token_count)
    )


def _should_trim_duplicate_prefix(token_overlap: TokenOverlap) -> bool:
    if token_overlap.token_count <= MIN_OVERLAP_TOKENS:
        return False
    if token_overlap.prefix_count < MIN_OVERLAP_TOKENS:
        return False
    return token_overlap.prefix_count < token_overlap.token_count


def _should_trim_duplicate_suffix(token_overlap: TokenOverlap) -> bool:
    if token_overlap.token_count <= MIN_OVERLAP_TOKENS:
        return False
    if token_overlap.suffix_count < MIN_OVERLAP_TOKENS:
        return False
    return token_overlap.suffix_count < token_overlap.token_count


def _trim_segment_tokens(
    segment: dict[str, Any],
    *,
    trim_prefix_tokens: int = 0,
    trim_suffix_tokens: int = 0,
) -> dict[str, Any] | None:
    token_spans = _token_spans(segment["transcript"])
    token_count = len(token_spans)
    keep_start_token = min(max(trim_prefix_tokens, 0), token_count)
    keep_end_token = max(min(token_count - max(trim_suffix_tokens, 0), token_count), 0)
    if token_count == 0 or keep_start_token >= keep_end_token:
        return None

    text_start = token_spans[keep_start_token][1]
    text_end = token_spans[keep_end_token - 1][2]
    transcript = segment["transcript"][text_start:text_end].strip()
    if not transcript:
        return None

    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(end - start, 0.0)
    trimmed_start = start + duration * (keep_start_token / token_count)
    trimmed_end = start + duration * (keep_end_token / token_count)

    trimmed = dict(segment)
    trimmed["start"] = round(trimmed_start, 3)
    trimmed["end"] = round(max(trimmed_start, trimmed_end), 3)
    trimmed["transcript"] = transcript
    trimmed["words"] = _trim_words(
        segment.get("words") or [],
        keep_start_token=keep_start_token,
        keep_end_token=keep_end_token,
        start=trimmed["start"],
        end=trimmed["end"],
    )
    return trimmed


def _trim_words(
    words: list[dict[str, Any]],
    *,
    keep_start_token: int,
    keep_end_token: int,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for word in words[keep_start_token:keep_end_token]:
        clipped = _clip_word(word, start=start, end=end)
        if clipped is not None:
            trimmed.append(clipped)
    return trimmed


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).lower(), match.start(), match.end())
        for match in re.finditer(r"[\w']+", text, flags=re.UNICODE)
    ]


def _clip_word(
    word: dict[str, Any],
    *,
    start: float | None,
    end: float | None,
) -> dict[str, Any] | None:
    word_start = float(word.get("start", 0.0))
    word_end = float(word.get("end", word_start))
    if start is not None and word_end < start:
        return None
    if end is not None and word_start > end:
        return None

    clipped_start = word_start if start is None else max(word_start, start)
    clipped_end = max(word_start, word_end) if end is None else min(max(word_start, word_end), end)
    clipped_start = round(clipped_start, 3)
    clipped_end = round(clipped_end, 3)
    if clipped_end < clipped_start:
        clipped_end = clipped_start

    return {
        "word": str(word.get("word", "")),
        "start": clipped_start,
        "end": clipped_end,
    }


def _audit_token_overlap(token_overlap: TokenOverlap) -> dict[str, Any]:
    return {
        "token_count": token_overlap.token_count,
        "covered_count": token_overlap.covered_count,
        "coverage": round(token_overlap.coverage, 3),
        "prefix_count": token_overlap.prefix_count,
        "suffix_count": token_overlap.suffix_count,
        "longest_run": token_overlap.longest_run,
    }


def render_stitch_audit_markdown(audit: dict[str, Any]) -> str:
    previous_chunk = _chunk_label(audit.get("previous_chunk"))
    next_chunk = _chunk_label(audit.get("next_chunk"))
    counts = audit.get("counts", {})
    lines = [
        "",
        f"## Chunk {previous_chunk} -> {next_chunk}",
        "",
        f"Method: {audit.get('method', 'fuzzy')}",
        f"Overlap: {audit['overlap_start_label']} -> {audit['overlap_end_label']}",
    ]
    if audit.get("safe_start_label") and audit.get("safe_end_label"):
        lines.append(f"Safe zone: {audit['safe_start_label']} -> {audit['safe_end_label']}")
    lines.extend(["", "Previous tail:"])
    lines.extend(_render_segment_lines(audit.get("previous_tail") or []))
    lines.extend(["", "Incoming head:"])
    for item in audit.get("incoming_head") or []:
        segment = item["segment"]
        original_segment = item.get("original_segment")
        if original_segment:
            lines.append(
                (
                    f"- {item['decision']} "
                    f"[{original_segment['start_label']} -> {original_segment['end_label']}] "
                    f"=> [{segment['start_label']} -> {segment['end_label']}] {segment['transcript']}"
                )
            )
        else:
            lines.append(
                f"- {item['decision']} [{segment['start_label']} -> {segment['end_label']}] {segment['transcript']}"
            )
    if not audit.get("incoming_head"):
        lines.append("- none")

    warning = audit.get("incoming_warning")
    if warning:
        lines.extend(
            [
                "",
                "Incoming warning:",
                f"- {warning.get('type', 'warning')} chunk={warning.get('chunk')} phrase={warning.get('phrase', '')}",
            ]
        )

    lines.extend(
        [
            "",
            "Decision:",
            (
                f"Dropped {counts.get('dropped_duplicates', 0)} duplicate segment(s) from overlap "
                f"({counts.get('dropped_exact_duplicates', 0)} exact, "
                f"{counts.get('dropped_overlap_duplicates', 0)} token-overlap), "
                f"dropped {counts.get('dropped_safe_zone', 0)} safe-zone edge segment(s), "
                f"trimmed {counts.get('trimmed_duplicate_prefix', 0)} prefix and "
                f"{counts.get('trimmed_duplicate_suffix', 0)} suffix duplicate segment(s), "
                f"kept {counts.get('kept', 0)} incoming segment(s) "
                f"({counts.get('kept_overlap', 0)} inside overlap, "
                f"{counts.get('kept_outside_overlap', 0)} outside overlap, "
                f"{counts.get('kept_low_confidence_overlap', 0)} low-confidence overlap)."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _render_segment_lines(segments: list[dict[str, Any]]) -> list[str]:
    if not segments:
        return ["- none"]
    return [
        f"- [{segment['start_label']} -> {segment['end_label']}] {segment['transcript']}"
        for segment in segments
    ]


def _audit_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": segment["start"],
        "end": segment["end"],
        "start_label": format_timestamp(segment["start"]),
        "end_label": format_timestamp(segment["end"]),
        "transcript": segment["transcript"],
    }


def _chunk_label(value: Any) -> str:
    if isinstance(value, int) and value >= 0:
        return f"{value:04d}"
    return "n/a"


def format_timestamp(seconds: float) -> str:
    millis_total = max(int(round(seconds * 1000)), 0)
    millis = millis_total % 1000
    total_seconds = millis_total // 1000
    secs = total_seconds % 60
    minutes_total = total_seconds // 60
    mins = minutes_total % 60
    hours = minutes_total // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def _normalize_for_compare(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()
