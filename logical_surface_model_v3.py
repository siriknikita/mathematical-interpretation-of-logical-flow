from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


VERSION_TAG = "v3"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


# =============================================================================
# Logical Surface Model v3
# =============================================================================
#
# Core idea:
#     Text is treated as a symbolic sheet.
#     Logical operators create local waves.
#     Logical scopes and relations create bounded bridge-fields.
#
# Compared with v2, this version adds:
#     1. clause-aware tokenization,
#     2. bounded relation scopes,
#     3. relation confidence scores,
#     4. list-or vs branching-or separation,
#     5. special handling for because inside if-then constructions,
#     6. thinner bridge geometry instead of broad interval flooding,
#     7. relation-energy normalization so relation_Z cannot swamp operator_Z,
#     8. extra diagnostics for overreach, confidence, and field balance.
#
# Dependencies:
#     pip install numpy matplotlib
#
# Run:
#     python logical_surface_model_v3.py
#     python logical_surface_model_v3.py --text "If A then B, but maybe not C."
#     python logical_surface_model_v3.py --file input.txt --save-prefix output/logical_surface
#
# With uv:
#     uv add numpy matplotlib
#     uv run logical_surface_model_v3.py --no-plots
# =============================================================================


@dataclass(frozen=True)
class Token:
    text: str
    index: int
    char_start: int
    char_end: int
    sentence_index: int
    clause_index: int


@dataclass(frozen=True)
class ClauseBoundary:
    token_after: int
    sentence_index: int
    clause_index_before: int
    reason: str


@dataclass(frozen=True)
class LogicalOperator:
    phrase: str
    family: str
    polarity: float
    strength: float
    width: float


@dataclass(frozen=True)
class OperatorEvent:
    phrase: str
    family: str
    token_start: int
    token_end: int
    center: float
    polarity: float
    strength: float
    width: float
    context_factor: float
    amplitude: float
    sentence_index: int
    clause_index: int


@dataclass(frozen=True)
class LogicalRelation:
    kind: str
    family: str
    source_start: int
    source_end: int
    target_start: int
    target_end: int
    source_center: float
    target_center: float
    operator_center: float
    amplitude: float
    raw_amplitude: float
    confidence: float
    width: float
    bridge_width: float
    sentence_index: int
    clause_index: int
    label: str
    evidence: str


@dataclass(frozen=True)
class SurfaceBundle:
    X: np.ndarray
    Y: np.ndarray
    operator_Z: np.ndarray
    relation_Z_raw: np.ndarray
    relation_Z: np.ndarray
    total_Z: np.ndarray
    positive_component: np.ndarray
    negative_component: np.ndarray
    relation_energy_scale: float


LOGICAL_OPERATORS: Tuple[LogicalOperator, ...] = (
    # Accumulation / continuation
    LogicalOperator("and", "conjunction", +0.35, 0.55, 4.0),
    LogicalOperator("also", "conjunction", +0.30, 0.50, 4.0),
    LogicalOperator("moreover", "conjunction", +0.45, 0.75, 5.0),
    LogicalOperator("furthermore", "conjunction", +0.45, 0.75, 5.0),
    LogicalOperator("as well as", "conjunction", +0.40, 0.65, 5.0),

    # Branching / alternatives
    LogicalOperator("or", "disjunction", +0.10, 0.85, 5.0),
    LogicalOperator("either", "disjunction", +0.10, 0.65, 5.0),
    LogicalOperator("either or", "disjunction", +0.15, 1.00, 6.0),
    LogicalOperator("alternatively", "disjunction", +0.10, 0.90, 6.0),

    # Inversion / denial
    LogicalOperator("not", "negation", -1.00, 1.00, 4.0),
    LogicalOperator("no", "negation", -0.90, 0.85, 4.0),
    LogicalOperator("never", "negation", -1.00, 1.10, 5.0),
    LogicalOperator("cannot", "negation", -1.00, 1.00, 4.0),
    LogicalOperator("can't", "negation", -1.00, 1.00, 4.0),
    LogicalOperator("without", "negation", -0.75, 0.80, 5.0),

    # Opposition / reversal
    LogicalOperator("but", "contrast", -0.70, 1.00, 6.0),
    LogicalOperator("however", "contrast", -0.75, 1.10, 7.0),
    LogicalOperator("although", "contrast", -0.55, 0.90, 7.0),
    LogicalOperator("though", "contrast", -0.45, 0.80, 6.0),
    LogicalOperator("nevertheless", "contrast", -0.80, 1.15, 7.0),
    LogicalOperator("despite", "contrast", -0.60, 0.95, 7.0),
    LogicalOperator("whereas", "contrast", -0.65, 0.95, 7.0),
    LogicalOperator("on the other hand", "contrast", -0.80, 1.20, 8.0),

    # Conditions
    LogicalOperator("if", "condition", +0.80, 1.00, 6.0),
    LogicalOperator("then", "condition", +0.70, 0.90, 6.0),
    LogicalOperator("when", "condition", +0.55, 0.75, 6.0),
    LogicalOperator("whenever", "condition", +0.65, 0.85, 6.0),
    LogicalOperator("unless", "condition", -0.85, 1.10, 7.0),
    LogicalOperator("provided that", "condition", +0.75, 1.00, 7.0),
    LogicalOperator("in case", "condition", +0.60, 0.85, 7.0),

    # Causality / inference
    LogicalOperator("because", "cause", +0.95, 1.10, 7.0),
    LogicalOperator("since", "cause", +0.65, 0.85, 6.0),
    LogicalOperator("therefore", "cause", +1.00, 1.15, 7.0),
    LogicalOperator("thus", "cause", +1.00, 1.10, 7.0),
    LogicalOperator("hence", "cause", +1.00, 1.10, 7.0),
    LogicalOperator("so", "cause", +0.75, 0.85, 6.0),
    LogicalOperator("as a result", "cause", +1.00, 1.20, 8.0),
    LogicalOperator("consequently", "cause", +1.00, 1.20, 8.0),
    LogicalOperator("due to", "cause", +0.75, 0.90, 7.0),

    # Evidence / examples
    LogicalOperator("for example", "evidence", +0.70, 0.80, 7.0),
    LogicalOperator("for instance", "evidence", +0.70, 0.80, 7.0),
    LogicalOperator("such as", "evidence", +0.60, 0.70, 6.0),
    LogicalOperator("e.g.", "evidence", +0.60, 0.70, 5.0),
    LogicalOperator("namely", "evidence", +0.65, 0.75, 6.0),

    # Epistemic uncertainty
    LogicalOperator("maybe", "uncertainty", -0.25, 0.65, 6.0),
    LogicalOperator("possibly", "uncertainty", -0.25, 0.65, 6.0),
    LogicalOperator("probably", "uncertainty", -0.15, 0.55, 6.0),
    LogicalOperator("might", "uncertainty", -0.25, 0.60, 5.0),
    LogicalOperator("could", "uncertainty", -0.20, 0.55, 5.0),
    LogicalOperator("seems", "uncertainty", -0.15, 0.50, 5.0),
)

FAMILY_ORDER: Tuple[str, ...] = (
    "conjunction",
    "disjunction",
    "negation",
    "contrast",
    "condition",
    "cause",
    "evidence",
    "uncertainty",
)

FAMILY_TO_Y: Dict[str, int] = {family: i for i, family in enumerate(FAMILY_ORDER)}

INTENSIFIERS = {
    "very",
    "really",
    "extremely",
    "strongly",
    "clearly",
    "definitely",
    "must",
    "always",
    "necessarily",
    "obviously",
}

HEDGES = {
    "maybe",
    "possibly",
    "probably",
    "somewhat",
    "kind",
    "sort",
    "might",
    "could",
    "seems",
    "apparently",
    "perhaps",
}

CONCLUSION_PHRASES = {"therefore", "thus", "hence", "as a result", "consequently", "so"}
CAUSAL_PHRASES = {"because", "since", "due to"}
CONTRAST_PHRASES = {
    "but",
    "however",
    "although",
    "though",
    "nevertheless",
    "despite",
    "whereas",
    "on the other hand",
}
NEGATION_PHRASES = {"not", "no", "never", "cannot", "can't", "without"}
UNCERTAINTY_PHRASES = {"maybe", "possibly", "probably", "might", "could", "seems"}
BRANCHING_PHRASES = {"or", "either", "either or", "alternatively"}
MAJOR_OPERATOR_FAMILIES = {"condition", "cause", "contrast"}


# =============================================================================
# Tokenization and structure helpers
# =============================================================================


def tokenize_with_structure(text: str) -> Tuple[List[Token], List[ClauseBoundary]]:
    """
    Tokenize text and assign sentence/clause indices.

    Punctuation is not returned as a token, but punctuation changes the
    sentence/clause counters:
        comma/colon/semicolon -> clause boundary
        period/question/exclamation/semicolon -> sentence boundary

    Semicolon is both a strong clause boundary and a sentence-like boundary.
    """
    pattern = r"e\.g\.|i\.e\.|[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?|[.!?;:,]"
    tokens: List[Token] = []
    boundaries: List[ClauseBoundary] = []
    sentence_index = 0
    clause_index = 0

    for match in re.finditer(pattern, text.lower()):
        raw = match.group(0)

        if raw in {",", ":"}:
            if tokens:
                boundaries.append(
                    ClauseBoundary(
                        token_after=len(tokens),
                        sentence_index=sentence_index,
                        clause_index_before=clause_index,
                        reason=raw,
                    )
                )
                clause_index += 1
            continue

        if raw in {".", "!", "?", ";"}:
            if tokens:
                boundaries.append(
                    ClauseBoundary(
                        token_after=len(tokens),
                        sentence_index=sentence_index,
                        clause_index_before=clause_index,
                        reason=raw,
                    )
                )
                sentence_index += 1
                clause_index += 1
            continue

        tokens.append(
            Token(
                text=raw,
                index=len(tokens),
                char_start=match.start(),
                char_end=match.end(),
                sentence_index=sentence_index,
                clause_index=clause_index,
            )
        )

    return tokens, boundaries


def phrase_to_tokens(phrase: str) -> Tuple[str, ...]:
    tokens, _ = tokenize_with_structure(phrase)
    return tuple(token.text for token in tokens)


OPERATOR_TOKEN_PATTERNS: Tuple[Tuple[Tuple[str, ...], LogicalOperator], ...] = tuple(
    sorted(
        ((phrase_to_tokens(op.phrase), op) for op in LOGICAL_OPERATORS),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def token_texts(tokens: Sequence[Token]) -> List[str]:
    return [token.text for token in tokens]


def span_center(start: int, end: int) -> float:
    if end <= start:
        return float(start)
    return (start + end - 1) / 2.0


def clamp_span(start: int, end: int, token_count: int) -> Tuple[int, int]:
    start = max(0, min(start, token_count))
    end = max(0, min(end, token_count))
    if end < start:
        start, end = end, start
    return start, end


def is_valid_span(start: int, end: int) -> bool:
    return end > start


def all_sentence_indices(tokens: Sequence[Token]) -> List[int]:
    return sorted(set(token.sentence_index for token in tokens))


def all_clause_indices(tokens: Sequence[Token]) -> List[int]:
    return sorted(set(token.clause_index for token in tokens))


def sentence_bounds(tokens: Sequence[Token], sentence_index: int) -> Tuple[int, int]:
    indices = [token.index for token in tokens if token.sentence_index == sentence_index]
    if not indices:
        return 0, 0
    return min(indices), max(indices) + 1


def clause_bounds(tokens: Sequence[Token], clause_index: int) -> Tuple[int, int]:
    indices = [token.index for token in tokens if token.clause_index == clause_index]
    if not indices:
        return 0, 0
    return min(indices), max(indices) + 1


def previous_sentence_bounds(tokens: Sequence[Token], sentence_index: int) -> Optional[Tuple[int, int]]:
    previous_indices = [idx for idx in all_sentence_indices(tokens) if idx < sentence_index]
    if not previous_indices:
        return None
    return sentence_bounds(tokens, previous_indices[-1])


def previous_clause_bounds(tokens: Sequence[Token], clause_index: int) -> Optional[Tuple[int, int]]:
    previous_indices = [idx for idx in all_clause_indices(tokens) if idx < clause_index]
    if not previous_indices:
        return None
    return clause_bounds(tokens, previous_indices[-1])


def next_clause_bounds(tokens: Sequence[Token], clause_index: int) -> Optional[Tuple[int, int]]:
    next_indices = [idx for idx in all_clause_indices(tokens) if idx > clause_index]
    if not next_indices:
        return None
    return clause_bounds(tokens, next_indices[0])


def next_operator_token_start(
    events: Sequence[OperatorEvent],
    *,
    after_token: int,
    sentence_index: int,
    allowed_families: Optional[set[str]] = None,
    excluded_phrases: Optional[set[str]] = None,
) -> Optional[int]:
    starts: List[int] = []
    for event in events:
        if event.sentence_index != sentence_index:
            continue
        if event.token_start <= after_token:
            continue
        if allowed_families is not None and event.family not in allowed_families:
            continue
        if excluded_phrases is not None and event.phrase in excluded_phrases:
            continue
        starts.append(event.token_start)
    if not starts:
        return None
    return min(starts)


def bounded_end(
    *,
    default_end: int,
    max_end: int,
    event: OperatorEvent,
    events: Sequence[OperatorEvent],
    max_tokens: Optional[int] = None,
    stop_at_major_operator: bool = True,
) -> int:
    end = min(default_end, max_end)

    if stop_at_major_operator:
        next_major = next_operator_token_start(
            events,
            after_token=event.token_end - 1,
            sentence_index=event.sentence_index,
            allowed_families=MAJOR_OPERATOR_FAMILIES,
            excluded_phrases={event.phrase},
        )
        if next_major is not None:
            end = min(end, next_major)

    if max_tokens is not None:
        end = min(end, event.token_end + max_tokens)

    return max(event.token_end, end)


# =============================================================================
# Operator detection
# =============================================================================


def local_context_factor(tokens: Sequence[Token], start: int, end: int, window: int = 4) -> float:
    texts = token_texts(tokens)
    left = max(0, start - window)
    right = min(len(texts), end + window)
    ctx = texts[left:right]

    factor = 1.0
    factor += 0.15 * sum(1 for token in ctx if token in INTENSIFIERS)
    factor -= 0.10 * sum(1 for token in ctx if token in HEDGES)
    return float(np.clip(factor, 0.55, 1.75))


def find_operator_events(text: str) -> Tuple[List[Token], List[ClauseBoundary], List[OperatorEvent]]:
    tokens, boundaries = tokenize_with_structure(text)
    texts = token_texts(tokens)
    events: List[OperatorEvent] = []

    occupied = np.zeros(len(tokens), dtype=bool)

    for pattern, op in OPERATOR_TOKEN_PATTERNS:
        n = len(pattern)
        if n == 0:
            continue

        for i in range(0, len(tokens) - n + 1):
            if occupied[i:i + n].any():
                continue

            if tuple(texts[i:i + n]) == pattern:
                center = i + (n - 1) / 2.0
                factor = local_context_factor(tokens, i, i + n)
                amplitude = op.polarity * op.strength * factor
                sentence_index = tokens[i].sentence_index
                clause_index = tokens[i].clause_index

                events.append(
                    OperatorEvent(
                        phrase=op.phrase,
                        family=op.family,
                        token_start=i,
                        token_end=i + n,
                        center=center,
                        polarity=op.polarity,
                        strength=op.strength,
                        width=op.width,
                        context_factor=factor,
                        amplitude=amplitude,
                        sentence_index=sentence_index,
                        clause_index=clause_index,
                    )
                )
                occupied[i:i + n] = True

    events.sort(key=lambda event: (event.token_start, event.token_end))
    return tokens, boundaries, events


def find_next_event(
    events: Iterable[OperatorEvent],
    *,
    after_token: int,
    sentence_index: int,
    phrase: Optional[str] = None,
    family: Optional[str] = None,
) -> Optional[OperatorEvent]:
    candidates = []
    for event in events:
        if event.sentence_index != sentence_index:
            continue
        if event.token_start < after_token:
            continue
        if phrase is not None and event.phrase != phrase:
            continue
        if family is not None and event.family != family:
            continue
        candidates.append(event)

    if not candidates:
        return None
    return min(candidates, key=lambda event: event.token_start)


def find_previous_event(
    events: Iterable[OperatorEvent],
    *,
    before_token: int,
    sentence_index: int,
    phrase: Optional[str] = None,
    family: Optional[str] = None,
) -> Optional[OperatorEvent]:
    candidates = []
    for event in events:
        if event.sentence_index != sentence_index:
            continue
        if event.token_start >= before_token:
            continue
        if phrase is not None and event.phrase != phrase:
            continue
        if family is not None and event.family != family:
            continue
        candidates.append(event)

    if not candidates:
        return None
    return max(candidates, key=lambda event: event.token_start)


def make_relation(
    *,
    kind: str,
    family: str,
    source_start: int,
    source_end: int,
    target_start: int,
    target_end: int,
    operator_center: float,
    raw_amplitude: float,
    confidence: float,
    width: float,
    bridge_width: float,
    sentence_index: int,
    clause_index: int,
    label: str,
    evidence: str,
    token_count: int,
) -> Optional[LogicalRelation]:
    source_start, source_end = clamp_span(source_start, source_end, token_count)
    target_start, target_end = clamp_span(target_start, target_end, token_count)

    if not is_valid_span(source_start, source_end):
        return None

    if not is_valid_span(target_start, target_end):
        target_start, target_end = source_start, source_end

    confidence = float(np.clip(confidence, 0.05, 1.0))
    amplitude = raw_amplitude * confidence

    return LogicalRelation(
        kind=kind,
        family=family,
        source_start=source_start,
        source_end=source_end,
        target_start=target_start,
        target_end=target_end,
        source_center=span_center(source_start, source_end),
        target_center=span_center(target_start, target_end),
        operator_center=operator_center,
        amplitude=amplitude,
        raw_amplitude=raw_amplitude,
        confidence=confidence,
        width=width,
        bridge_width=bridge_width,
        sentence_index=sentence_index,
        clause_index=clause_index,
        label=label,
        evidence=evidence,
    )


def is_list_or(tokens: Sequence[Token], events: Sequence[OperatorEvent], event: OperatorEvent) -> bool:
    """
    Distinguish list-disjunction from genuine branching.

    Example:
        "maybe, possibly, or not" is a list of markers, not a semantic fork.
    """
    if event.phrase not in {"or", "either or"}:
        return False

    left_start = max(0, event.token_start - 3)
    right_end = min(len(tokens), event.token_end + 3)
    local_words = {token.text for token in tokens[left_start:right_end]}

    if local_words & (UNCERTAINTY_PHRASES | NEGATION_PHRASES):
        return True

    nearby_events = [
        other for other in events
        if other is not event and abs(other.center - event.center) <= 3.0
    ]
    nearby_non_branching = [
        other for other in nearby_events
        if other.family in {"uncertainty", "negation", "conjunction"}
    ]
    return len(nearby_non_branching) >= 1


def infer_logical_relations(tokens: Sequence[Token], events: Sequence[OperatorEvent]) -> List[LogicalRelation]:
    """
    Convert isolated operator events into bounded approximate logical relations.

    This is still heuristic, but v3 intentionally avoids v2's main failure mode:
    relation scopes should not cover almost the whole text unless the text is
    actually a long single relation.
    """
    relations: List[LogicalRelation] = []
    token_count = len(tokens)

    for event in events:
        sent_start, sent_end = sentence_bounds(tokens, event.sentence_index)
        clause_start, clause_end = clause_bounds(tokens, event.clause_index)

        if event.phrase == "if":
            then_event = find_next_event(
                events,
                after_token=event.token_end,
                sentence_index=event.sentence_index,
                phrase="then",
            )

            if then_event is not None:
                source_start = event.token_end
                source_end = then_event.token_start
                target_start = then_event.token_end
                target_end = bounded_end(
                    default_end=sent_end,
                    max_end=sent_end,
                    event=then_event,
                    events=events,
                    max_tokens=14,
                    stop_at_major_operator=True,
                )
                confidence = 0.96
                label = "if_scope -> then_scope"
                evidence = "explicit_if_then"
                raw_amplitude = +0.85 * event.context_factor
                width = 4.5
                bridge_width = 0.34
            else:
                source_start = event.token_end
                source_end = bounded_end(
                    default_end=clause_end,
                    max_end=sent_end,
                    event=event,
                    events=events,
                    max_tokens=12,
                    stop_at_major_operator=True,
                )
                target_start = source_start
                target_end = source_end
                confidence = 0.50
                label = "bounded_if_scope"
                evidence = "if_without_then"
                raw_amplitude = +0.35 * event.context_factor
                width = 3.8
                bridge_width = 0.32

            relation = make_relation(
                kind="implication" if then_event is not None else "conditional_scope",
                family="condition",
                source_start=source_start,
                source_end=source_end,
                target_start=target_start,
                target_end=target_end,
                operator_center=event.center,
                raw_amplitude=raw_amplitude,
                confidence=confidence,
                width=width,
                bridge_width=bridge_width,
                sentence_index=event.sentence_index,
                clause_index=event.clause_index,
                label=label,
                evidence=evidence,
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in CAUSAL_PHRASES:
            previous_if = find_previous_event(
                events,
                before_token=event.token_start,
                sentence_index=event.sentence_index,
                phrase="if",
            )
            next_then = find_next_event(
                events,
                after_token=event.token_end,
                sentence_index=event.sentence_index,
                phrase="then",
            )

            if previous_if is not None and next_then is not None and event.token_start < next_then.token_start:
                # because inside an if-then construction should support the local condition,
                # not spill across the entire then-consequence.
                source_start = event.token_end
                source_end = next_then.token_start
                target_start = previous_if.token_end
                target_end = event.token_start
                confidence = 0.82
                evidence = "because_inside_if_then"
                label = f"{event.phrase}_local_supports_if_scope"
                raw_amplitude = +0.62 * event.context_factor
                width = 3.7
                bridge_width = 0.32
            elif event.token_start > clause_start:
                source_start = event.token_end
                source_end = bounded_end(
                    default_end=clause_end,
                    max_end=sent_end,
                    event=event,
                    events=events,
                    max_tokens=12,
                    stop_at_major_operator=True,
                )
                target_start = clause_start
                target_end = event.token_start
                confidence = 0.76
                evidence = "because_after_claim_same_clause"
                label = f"{event.phrase}_cause -> local_effect"
                raw_amplitude = +0.72 * event.context_factor
                width = 4.2
                bridge_width = 0.34
            else:
                source_start = event.token_end
                source_end = bounded_end(
                    default_end=clause_end,
                    max_end=sent_end,
                    event=event,
                    events=events,
                    max_tokens=10,
                    stop_at_major_operator=True,
                )
                following_clause = next_clause_bounds(tokens, event.clause_index)
                if following_clause is not None:
                    target_start, target_end = following_clause
                else:
                    target_start = source_start
                    target_end = source_end
                confidence = 0.58
                evidence = "fronted_causal_scope"
                label = f"{event.phrase}_fronted_cause"
                raw_amplitude = +0.60 * event.context_factor
                width = 3.8
                bridge_width = 0.32

            relation = make_relation(
                kind="causal_support",
                family="cause",
                source_start=source_start,
                source_end=source_end,
                target_start=target_start,
                target_end=target_end,
                operator_center=event.center,
                raw_amplitude=raw_amplitude,
                confidence=confidence,
                width=width,
                bridge_width=bridge_width,
                sentence_index=event.sentence_index,
                clause_index=event.clause_index,
                label=label,
                evidence=evidence,
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in CONCLUSION_PHRASES:
            previous_clause = previous_clause_bounds(tokens, event.clause_index)
            previous_sentence = previous_sentence_bounds(tokens, event.sentence_index)

            if event.token_start > clause_start:
                source_start = clause_start
                source_end = event.token_start
                confidence = 0.78
                evidence = "conclusion_same_clause"
            elif previous_clause is not None:
                source_start, source_end = previous_clause
                confidence = 0.72
                evidence = "conclusion_previous_clause"
            elif previous_sentence is not None:
                source_start, source_end = previous_sentence
                confidence = 0.70
                evidence = "conclusion_previous_sentence"
            else:
                source_start, source_end = sent_start, event.token_start
                confidence = 0.45
                evidence = "conclusion_weak_previous_context"

            target_start = event.token_end
            target_end = bounded_end(
                default_end=clause_end,
                max_end=sent_end,
                event=event,
                events=events,
                max_tokens=14,
                stop_at_major_operator=True,
            )

            relation = make_relation(
                kind="conclusion",
                family="cause",
                source_start=source_start,
                source_end=source_end,
                target_start=target_start,
                target_end=target_end,
                operator_center=event.center,
                raw_amplitude=+0.90 * event.context_factor,
                confidence=confidence,
                width=4.8,
                bridge_width=0.36,
                sentence_index=event.sentence_index,
                clause_index=event.clause_index,
                label=f"premises -> {event.phrase}_conclusion",
                evidence=evidence,
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in CONTRAST_PHRASES:
            previous_clause = previous_clause_bounds(tokens, event.clause_index)
            previous_sentence = previous_sentence_bounds(tokens, event.sentence_index)

            if event.token_start > clause_start:
                source_start = clause_start
                source_end = event.token_start
                confidence = 0.86
                evidence = "contrast_same_clause"
            elif previous_clause is not None and tokens[previous_clause[0]].sentence_index == event.sentence_index:
                source_start, source_end = previous_clause
                confidence = 0.78
                evidence = "contrast_previous_clause"
            elif previous_sentence is not None:
                source_start, source_end = previous_sentence
                confidence = 0.72
                evidence = "contrast_previous_sentence"
            else:
                source_start = sent_start
                source_end = event.token_start
                confidence = 0.45
                evidence = "contrast_weak_previous_context"

            target_start = event.token_end
            target_end = bounded_end(
                default_end=sent_end if event.phrase in {"however", "nevertheless", "but"} else clause_end,
                max_end=sent_end,
                event=event,
                events=events,
                max_tokens=16,
                stop_at_major_operator=False,
            )

            relation = make_relation(
                kind="contrast",
                family="contrast",
                source_start=source_start,
                source_end=source_end,
                target_start=target_start,
                target_end=target_end,
                operator_center=event.center,
                raw_amplitude=-0.72 * event.context_factor,
                confidence=confidence,
                width=4.4,
                bridge_width=0.36,
                sentence_index=event.sentence_index,
                clause_index=event.clause_index,
                label=f"left_claim <-> {event.phrase}_right_claim",
                evidence=evidence,
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in BRANCHING_PHRASES:
            list_or = is_list_or(tokens, events, event)
            kind = "list_disjunction" if list_or else "branching"
            raw_amplitude = +0.18 * event.context_factor if list_or else +0.45 * event.context_factor
            confidence = 0.82 if list_or else 0.68
            source_floor = sent_start if list_or else clause_start
            target_ceiling = sent_end if list_or else clause_end
            relation = make_relation(
                kind=kind,
                family="disjunction",
                source_start=max(source_floor, event.token_start - 3),
                source_end=event.token_start,
                target_start=event.token_end,
                target_end=min(target_ceiling, event.token_end + 3),
                operator_center=event.center,
                raw_amplitude=raw_amplitude,
                confidence=confidence,
                width=2.8 if list_or else 3.8,
                bridge_width=0.28,
                sentence_index=event.sentence_index,
                clause_index=event.clause_index,
                label="list_left or list_right" if list_or else "branch_left or branch_right",
                evidence="uncertainty_or_negation_list" if list_or else "alternative_branching",
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in NEGATION_PHRASES or event.phrase == "unless":
            scope_end = bounded_end(
                default_end=clause_end,
                max_end=sent_end,
                event=event,
                events=events,
                max_tokens=6,
                stop_at_major_operator=True,
            )
            relation = make_relation(
                kind="negation_scope",
                family="negation",
                source_start=event.token_start,
                source_end=scope_end,
                target_start=event.token_start,
                target_end=scope_end,
                operator_center=event.center,
                raw_amplitude=-0.54 * event.context_factor,
                confidence=0.88,
                width=2.8,
                bridge_width=0.30,
                sentence_index=event.sentence_index,
                clause_index=event.clause_index,
                label=f"{event.phrase}_bounded_scope",
                evidence="local_negation_scope",
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in UNCERTAINTY_PHRASES:
            scope_end = bounded_end(
                default_end=clause_end,
                max_end=sent_end,
                event=event,
                events=events,
                max_tokens=5,
                stop_at_major_operator=True,
            )
            relation = make_relation(
                kind="uncertainty_scope",
                family="uncertainty",
                source_start=event.token_start,
                source_end=scope_end,
                target_start=event.token_start,
                target_end=scope_end,
                operator_center=event.center,
                raw_amplitude=-0.30 * event.context_factor,
                confidence=0.78,
                width=2.8,
                bridge_width=0.30,
                sentence_index=event.sentence_index,
                clause_index=event.clause_index,
                label=f"{event.phrase}_bounded_scope",
                evidence="local_uncertainty_scope",
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

    relations.sort(key=lambda relation: (relation.source_start, relation.target_start, relation.kind))
    return relations


# =============================================================================
# Field construction
# =============================================================================


def mexican_hat_wave(distance: np.ndarray, width: float) -> np.ndarray:
    safe_width = max(width, 1e-6)
    x = distance / safe_width
    return (1.0 - x ** 2) * np.exp(-0.5 * x ** 2)


def gaussian_kernel(distance: np.ndarray, width: float) -> np.ndarray:
    safe_width = max(width, 1e-6)
    return np.exp(-0.5 * (distance / safe_width) ** 2)


def interval_kernel(X: np.ndarray, start: float, end: float, edge_width: float) -> np.ndarray:
    if end < start:
        start, end = end, start

    if abs(end - start) < 1e-6:
        return gaussian_kernel(X - start, edge_width)

    outside_left = np.maximum(start - X, 0.0)
    outside_right = np.maximum(X - end, 0.0)
    outside = outside_left + outside_right

    kernel = np.exp(-0.5 * (outside / max(edge_width, 1e-6)) ** 2)
    inside = (X >= start) & (X <= end)
    kernel[inside] = 1.0
    return kernel


def raised_cosine_between(X: np.ndarray, start: float, end: float) -> np.ndarray:
    if end < start:
        start, end = end, start
    length = max(end - start, 1e-6)
    phase = np.pi * (X - start) / length
    clipped = np.clip(phase, 0.0, np.pi)
    shape = np.sin(clipped)
    shape[(X < start) | (X > end)] = 0.0
    return shape


def bounded_bridge_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    start_x: float,
    end_x: float,
    family_y: float,
    x_edge_width: float,
    y_width: float,
) -> np.ndarray:
    """
    Thin bridge field between two text regions.

    v2 used broad interval fields; v3 uses a narrow bridge that fades outside
    the source-target corridor and stays thin across logical families.
    """
    if end_x < start_x:
        start_x, end_x = end_x, start_x

    length = max(end_x - start_x, 1e-6)
    x_gate = interval_kernel(X, start_x, end_x, edge_width=x_edge_width)
    arch = 0.35 + 0.65 * raised_cosine_between(X, start_x, end_x)
    taper = gaussian_kernel((X - (start_x + end_x) / 2.0) / max(length, 1.0), 0.42)
    y_gate = gaussian_kernel(Y - family_y, y_width)
    return x_gate * arch * taper * y_gate


def scope_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    start_x: float,
    end_x: float,
    family_y: float,
    x_edge_width: float,
    y_width: float,
) -> np.ndarray:
    interval = interval_kernel(X, start_x, end_x, edge_width=x_edge_width)
    y_gate = gaussian_kernel(Y - family_y, y_width)
    return interval * y_gate


def normalize_relation_energy(
    operator_Z: np.ndarray,
    relation_Z_raw: np.ndarray,
    max_relation_to_operator_ratio: float,
) -> Tuple[np.ndarray, float]:
    operator_energy = float(np.mean(operator_Z ** 2))
    relation_energy = float(np.mean(relation_Z_raw ** 2))

    if relation_energy <= 1e-12:
        return relation_Z_raw.copy(), 1.0

    if operator_energy <= 1e-12:
        return relation_Z_raw.copy(), 1.0

    allowed_relation_energy = operator_energy * max_relation_to_operator_ratio
    if relation_energy <= allowed_relation_energy:
        return relation_Z_raw.copy(), 1.0

    scale = math.sqrt(allowed_relation_energy / relation_energy)
    return relation_Z_raw * scale, float(scale)


def build_surface(
    tokens: Sequence[Token],
    events: Sequence[OperatorEvent],
    relations: Sequence[LogicalRelation],
    x_resolution: int = 280,
    y_resolution: int = 140,
    operator_cross_family_width: float = 0.55,
    relation_cross_family_width: float = 0.42,
    max_relation_to_operator_ratio: float = 1.25,
) -> SurfaceBundle:
    if len(tokens) == 0:
        x = np.linspace(0, 1, x_resolution)
    else:
        x = np.linspace(0, max(1, len(tokens) - 1), x_resolution)

    y = np.linspace(0, len(FAMILY_ORDER) - 1, y_resolution)
    X, Y = np.meshgrid(x, y)

    operator_Z = np.zeros_like(X, dtype=float)
    relation_Z_raw = np.zeros_like(X, dtype=float)

    for event in events:
        family_y = FAMILY_TO_Y[event.family]
        dx = X - event.center
        dy = Y - family_y

        longitudinal_wave = mexican_hat_wave(dx, event.width)
        family_kernel = gaussian_kernel(dy, operator_cross_family_width)
        operator_Z += event.amplitude * longitudinal_wave * family_kernel

    for relation in relations:
        family_y = FAMILY_TO_Y[relation.family]
        source_center = relation.source_center
        target_center = relation.target_center
        start_x = min(source_center, target_center, relation.operator_center)
        end_x = max(source_center, target_center, relation.operator_center)

        if relation.kind in {"negation_scope", "uncertainty_scope", "conditional_scope", "list_disjunction"}:
            kernel = scope_kernel(
                X,
                Y,
                start_x=relation.source_start,
                end_x=max(relation.source_end - 1, relation.source_start),
                family_y=family_y,
                x_edge_width=relation.width,
                y_width=relation_cross_family_width,
            )
        else:
            kernel = bounded_bridge_kernel(
                X,
                Y,
                start_x=start_x,
                end_x=end_x,
                family_y=family_y,
                x_edge_width=relation.width,
                y_width=relation.bridge_width,
            )

        if relation.kind == "branching":
            # Branching should look fork-like/turbulent, not like a causal ridge.
            oscillation = 0.75 + 0.25 * np.cos((X - relation.operator_center) / max(relation.width, 1e-6))
            relation_Z_raw += relation.amplitude * kernel * oscillation
        elif relation.kind == "contrast":
            # Contrast creates a negative bridge plus a small opposite-edge ripple.
            ripple = 1.0 + 0.12 * np.sin((X - relation.operator_center) / max(relation.width, 1e-6))
            relation_Z_raw += relation.amplitude * kernel * ripple
        else:
            relation_Z_raw += relation.amplitude * kernel

    relation_Z, relation_energy_scale = normalize_relation_energy(
        operator_Z,
        relation_Z_raw,
        max_relation_to_operator_ratio=max_relation_to_operator_ratio,
    )
    total_Z = operator_Z + relation_Z
    positive_component = np.clip(operator_Z, 0.0, None) + np.clip(relation_Z, 0.0, None)
    negative_component = np.clip(-operator_Z, 0.0, None) + np.clip(-relation_Z, 0.0, None)

    return SurfaceBundle(
        X=X,
        Y=Y,
        operator_Z=operator_Z,
        relation_Z_raw=relation_Z_raw,
        relation_Z=relation_Z,
        total_Z=total_Z,
        positive_component=positive_component,
        negative_component=negative_component,
        relation_energy_scale=relation_energy_scale,
    )


# =============================================================================
# Metrics
# =============================================================================


def compute_max_relation_overlap(relations: Sequence[LogicalRelation]) -> int:
    points: List[Tuple[int, int]] = []
    for relation in relations:
        start = min(relation.source_start, relation.target_start)
        end = max(relation.source_end, relation.target_end)
        if end <= start:
            continue
        points.append((start, +1))
        points.append((end, -1))

    if not points:
        return 0

    points.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    max_active = 0
    for _, delta in points:
        active += delta
        max_active = max(max_active, active)
    return max_active


def compute_scope_coverage(token_count: int, relations: Sequence[LogicalRelation]) -> float:
    if token_count <= 0 or not relations:
        return 0.0

    covered = np.zeros(token_count, dtype=bool)
    for relation in relations:
        source_start, source_end = clamp_span(relation.source_start, relation.source_end, token_count)
        target_start, target_end = clamp_span(relation.target_start, relation.target_end, token_count)
        covered[source_start:source_end] = True
        covered[target_start:target_end] = True

    return float(np.mean(covered))


def compute_bridge_coverage(token_count: int, relations: Sequence[LogicalRelation]) -> float:
    if token_count <= 0 or not relations:
        return 0.0

    covered = np.zeros(token_count, dtype=bool)
    for relation in relations:
        if relation.kind in {"negation_scope", "uncertainty_scope", "conditional_scope", "list_disjunction"}:
            start, end = relation.source_start, relation.source_end
        else:
            start = int(max(0, math.floor(min(relation.source_center, relation.target_center))))
            end = int(min(token_count, math.ceil(max(relation.source_center, relation.target_center)) + 1))
        covered[start:end] = True

    return float(np.mean(covered))


def compute_metrics(
    tokens: Sequence[Token],
    events: Sequence[OperatorEvent],
    relations: Sequence[LogicalRelation],
    surface: SurfaceBundle,
) -> Dict[str, float]:
    token_count = len(tokens)
    sentence_count = len(all_sentence_indices(tokens))
    clause_count = len(all_clause_indices(tokens))
    per_100 = 100.0 / max(1, token_count)

    Z = surface.total_Z
    positive_density = float(np.sum(surface.positive_component) / surface.positive_component.size)
    negative_density = float(np.sum(surface.negative_component) / surface.negative_component.size)

    gy, gx = np.gradient(Z)
    roughness = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    tension = float(np.mean(np.sqrt(surface.positive_component * surface.negative_component)))

    relation_counts: Dict[str, int] = {}
    for relation in relations:
        relation_counts[relation.kind] = relation_counts.get(relation.kind, 0) + 1

    family_counts: Dict[str, int] = {}
    for event in events:
        family_counts[event.family] = family_counts.get(event.family, 0) + 1

    confidence_values = [relation.confidence for relation in relations]
    avg_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
    min_confidence = float(np.min(confidence_values)) if confidence_values else 0.0

    conclusion_force = sum(
        max(0.0, relation.amplitude)
        for relation in relations
        if relation.kind == "conclusion"
    ) * per_100
    contrast_relation_pressure = sum(
        abs(relation.amplitude)
        for relation in relations
        if relation.kind == "contrast"
    ) * per_100
    uncertainty_scope_pressure = sum(
        abs(relation.amplitude)
        for relation in relations
        if relation.kind == "uncertainty_scope"
    ) * per_100
    negation_scope_pressure = sum(
        abs(relation.amplitude)
        for relation in relations
        if relation.kind == "negation_scope"
    ) * per_100

    operator_energy = float(np.mean(surface.operator_Z ** 2))
    relation_energy_raw = float(np.mean(surface.relation_Z_raw ** 2))
    relation_energy = float(np.mean(surface.relation_Z ** 2))
    total_energy = float(np.mean(surface.total_Z ** 2))

    semantic_stability = 1.0 / (
        1.0
        + roughness
        + tension
        + negative_density
        + 0.08 * contrast_relation_pressure
        + 0.08 * uncertainty_scope_pressure
        + 0.04 * max(0.0, relation_energy / max(operator_energy, 1e-12) - 1.0)
    )

    return {
        "token_count": float(token_count),
        "sentence_count": float(sentence_count),
        "clause_count": float(clause_count),
        "operator_count": float(len(events)),
        "relation_count": float(len(relations)),
        "operator_density_per_100_tokens": float(len(events) * per_100),
        "relation_density_per_100_tokens": float(len(relations) * per_100),

        "positive_wave_density": positive_density,
        "negative_wave_density": negative_density,
        "logical_balance_ratio": float(positive_density / max(negative_density, 1e-12)),
        "net_logical_elevation": float(np.mean(Z)),
        "surface_energy_density": total_energy,
        "surface_roughness_density": roughness,
        "logical_tension_density": tension,

        "operator_field_energy_density": operator_energy,
        "relation_field_raw_energy_density": relation_energy_raw,
        "relation_field_energy_density": relation_energy,
        "relation_to_operator_energy_ratio_raw": float(relation_energy_raw / max(operator_energy, 1e-12)),
        "relation_to_operator_energy_ratio": float(relation_energy / max(operator_energy, 1e-12)),
        "relation_energy_scale": float(surface.relation_energy_scale),

        "branching_density": float(family_counts.get("disjunction", 0) * per_100),
        "contrast_pressure": float(family_counts.get("contrast", 0) * per_100),
        "causal_inference_density": float(
            (family_counts.get("cause", 0) + family_counts.get("condition", 0)) * per_100
        ),
        "uncertainty_density": float(family_counts.get("uncertainty", 0) * per_100),

        "implication_relation_density": float(relation_counts.get("implication", 0) * per_100),
        "causal_relation_density": float(relation_counts.get("causal_support", 0) * per_100),
        "contrast_relation_density": float(relation_counts.get("contrast", 0) * per_100),
        "branching_relation_density": float(relation_counts.get("branching", 0) * per_100),
        "list_disjunction_density": float(relation_counts.get("list_disjunction", 0) * per_100),
        "negation_scope_density": float(relation_counts.get("negation_scope", 0) * per_100),
        "uncertainty_scope_density": float(relation_counts.get("uncertainty_scope", 0) * per_100),

        "conclusion_force": float(conclusion_force),
        "contrast_relation_pressure": float(contrast_relation_pressure),
        "negation_scope_pressure": float(negation_scope_pressure),
        "uncertainty_scope_pressure": float(uncertainty_scope_pressure),
        "argumentative_depth": float(compute_max_relation_overlap(relations)),
        "scope_coverage_ratio": float(compute_scope_coverage(token_count, relations)),
        "bridge_coverage_ratio": float(compute_bridge_coverage(token_count, relations)),
        "average_relation_confidence": avg_confidence,
        "minimum_relation_confidence": min_confidence,
        "semantic_stability": float(semantic_stability),
    }


# =============================================================================
# Output helpers
# =============================================================================


def format_span(tokens: Sequence[Token], start: int, end: int, max_tokens: int = 10) -> str:
    start, end = clamp_span(start, end, len(tokens))
    texts = [token.text for token in tokens[start:end]]
    if len(texts) > max_tokens:
        texts = texts[:max_tokens] + ["..."]
    return " ".join(texts)


def print_events(events: Sequence[OperatorEvent]) -> None:
    if not events:
        print("No logical operators were detected.")
        return

    print("\nDetected logical-operator events:")
    print("-" * 124)
    print(
        f"{'phrase':<22} {'family':<14} {'sentence':>8} {'clause':>7} {'tokens':<12} "
        f"{'amplitude':>10} {'context':>9}"
    )
    print("-" * 124)

    for event in events:
        token_span = f"{event.token_start}:{event.token_end}"
        print(
            f"{event.phrase:<22} {event.family:<14} {event.sentence_index:>8} "
            f"{event.clause_index:>7} {token_span:<12} {event.amplitude:>10.3f} "
            f"{event.context_factor:>9.3f}"
        )


def print_relations(tokens: Sequence[Token], relations: Sequence[LogicalRelation]) -> None:
    if not relations:
        print("\nNo logical relations were inferred.")
        return

    print("\nInferred bounded logical relations:")
    print("-" * 172)
    print(
        f"{'kind':<22} {'family':<12} {'sent':>5} {'clause':>6} {'amp':>8} "
        f"{'conf':>6} {'source span':<34} {'target span':<34} {'label':<30} {'evidence':<26}"
    )
    print("-" * 172)

    for relation in relations:
        source = format_span(tokens, relation.source_start, relation.source_end)
        target = format_span(tokens, relation.target_start, relation.target_end)
        print(
            f"{relation.kind:<22} {relation.family:<12} {relation.sentence_index:>5} "
            f"{relation.clause_index:>6} {relation.amplitude:>8.3f} {relation.confidence:>6.2f} "
            f"{source:<34} {target:<34} {relation.label:<30} {relation.evidence:<26}"
        )


def print_metrics(metrics: Dict[str, float]) -> None:
    print("\nSurface/text metrics:")
    print("-" * 124)

    integer_keys = {
        "token_count",
        "sentence_count",
        "clause_count",
        "operator_count",
        "relation_count",
        "argumentative_depth",
    }

    for key, value in metrics.items():
        if key in integer_keys:
            print(f"{key:<48} {int(value)}")
        else:
            print(f"{key:<48} {value:.6f}")


def interpret_metrics(metrics: Dict[str, float]) -> str:
    balance = metrics["logical_balance_ratio"]
    tension = metrics["logical_tension_density"]
    relation_ratio = metrics["relation_to_operator_energy_ratio"]
    raw_relation_ratio = metrics["relation_to_operator_energy_ratio_raw"]
    stability = metrics["semantic_stability"]
    bridge_coverage = metrics["bridge_coverage_ratio"]
    avg_confidence = metrics["average_relation_confidence"]

    if balance > 1.25:
        balance_text = "constructive/inferential forces dominate"
    elif balance < 0.80:
        balance_text = "negation/contrast/uncertainty forces dominate"
    else:
        balance_text = "constructive and destabilizing forces are relatively balanced"

    if tension > 0.10:
        tension_text = "high local logical tension"
    elif tension > 0.04:
        tension_text = "moderate local logical tension"
    else:
        tension_text = "low local logical tension"

    if relation_ratio > 1.10:
        relation_text = "relations are strong but energy-capped"
    elif relation_ratio > 0.35:
        relation_text = "relations substantially shape the surface"
    else:
        relation_text = "the surface is mostly driven by local operators"

    if raw_relation_ratio > relation_ratio * 1.50:
        normalization_text = "raw relation overreach was detected and normalized"
    else:
        normalization_text = "no severe relation overreach was detected"

    if bridge_coverage > 0.80:
        coverage_text = "broad text coverage"
    elif bridge_coverage > 0.45:
        coverage_text = "moderate text coverage"
    else:
        coverage_text = "localized bridge coverage"

    if avg_confidence > 0.80:
        confidence_text = "high average relation confidence"
    elif avg_confidence > 0.60:
        confidence_text = "moderate average relation confidence"
    else:
        confidence_text = "low average relation confidence"

    if stability > 0.80:
        stability_text = "high semantic stability"
    elif stability > 0.55:
        stability_text = "moderate semantic stability"
    else:
        stability_text = "low semantic stability"

    return (
        f"Interpretation: {balance_text}; {tension_text}; {relation_text}; "
        f"{normalization_text}; {coverage_text}; {confidence_text}; {stability_text}."
    )


# =============================================================================
# Plotting
# =============================================================================


def select_field(surface: SurfaceBundle, field: str) -> np.ndarray:
    if field == "operator":
        return surface.operator_Z
    if field == "relation_raw":
        return surface.relation_Z_raw
    if field == "relation":
        return surface.relation_Z
    if field == "total":
        return surface.total_Z
    raise ValueError("field must be one of: operator, relation_raw, relation, total")


def plot_surface(
    surface: SurfaceBundle,
    field: str = "total",
    title: str = "Logical Surface of Text",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    Z = select_field(surface, field)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    plotted = ax.plot_surface(
        surface.X,
        surface.Y,
        Z,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )

    ax.set_title(title, pad=20)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Logical family")
    ax.set_zlabel("Logical elevation")
    ax.set_yticks(list(FAMILY_TO_Y.values()))
    ax.set_yticklabels(FAMILY_ORDER)

    fig.colorbar(plotted, shrink=0.55, aspect=12, pad=0.08, label="Elevation")
    fig.subplots_adjust(left=0.04, right=0.86, bottom=0.08, top=0.92)

    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_heatmap(
    surface: SurfaceBundle,
    field: str = "total",
    title: str = "Logical Surface Heatmap",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    Z = select_field(surface, field)

    fig, ax = plt.subplots(figsize=(13, 5))
    plotted = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[surface.X.min(), surface.X.max(), surface.Y.min(), surface.Y.max()],
    )
    ax.set_title(title)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Logical family")
    ax.set_yticks(list(FAMILY_TO_Y.values()))
    ax.set_yticklabels(FAMILY_ORDER)

    fig.colorbar(plotted, ax=ax, label="Logical elevation")
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.15, top=0.90)

    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Analysis entry point
# =============================================================================


def operator_catalog_signature() -> Dict[str, object]:
    payload = "|".join(
        f"{op.phrase}:{op.family}:{op.polarity}:{op.strength}:{op.width}"
        for op in LOGICAL_OPERATORS
    )
    return {
        "count": len(LOGICAL_OPERATORS),
        "families": list(FAMILY_ORDER),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def generate_run_id(text: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
    suffix = uuid.uuid4().hex[:4]
    return f"{stamp}_{digest}_{suffix}"


def prepare_run_directory(output_root: Path, run_id: str) -> Path:
    run_dir = output_root / VERSION_TAG / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_params(
    run_dir: Path,
    *,
    run_id: str,
    text: str,
    config: Dict[str, object],
) -> Path:
    params = {
        "run_id": run_id,
        "version": VERSION_TAG,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "input": {
            "char_count": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "preview": text.strip()[:200],
        },
        "config": config,
        "operator_catalog": operator_catalog_signature(),
        "intensifiers": sorted(INTENSIFIERS),
        "hedges": sorted(HEDGES),
    }
    params_path = run_dir / "params.json"
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    (run_dir / "input_text.txt").write_text(text, encoding="utf-8")
    return params_path


def write_run_metrics(run_dir: Path, metrics: Dict[str, float]) -> Path:
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def analyze_text(
    text: str,
    *,
    show_plots: bool = True,
    save: bool = True,
    output_root: Optional[Path] = None,
    run_id: Optional[str] = None,
    x_resolution: int = 280,
    y_resolution: int = 140,
    operator_cross_family_width: float = 0.55,
    relation_cross_family_width: float = 0.42,
    max_relation_to_operator_ratio: float = 1.25,
) -> Dict[str, object]:
    tokens, boundaries, events = find_operator_events(text)
    relations = infer_logical_relations(tokens, events)
    surface = build_surface(
        tokens,
        events,
        relations,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        operator_cross_family_width=operator_cross_family_width,
        relation_cross_family_width=relation_cross_family_width,
        max_relation_to_operator_ratio=max_relation_to_operator_ratio,
    )
    metrics = compute_metrics(tokens, events, relations, surface)

    print(
        f"Tokenized text into {len(tokens)} tokens across "
        f"{len(all_sentence_indices(tokens))} sentence block(s) and "
        f"{len(all_clause_indices(tokens))} clause block(s)."
    )
    print_events(events)
    print_relations(tokens, relations)
    print_metrics(metrics)
    print("\n" + interpret_metrics(metrics))

    run_dir: Optional[Path] = None
    if save:
        root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
        resolved_run_id = run_id or generate_run_id(text)
        run_dir = prepare_run_directory(root, resolved_run_id)
        config = {
            "x_resolution": x_resolution,
            "y_resolution": y_resolution,
            "operator_cross_family_width": operator_cross_family_width,
            "relation_cross_family_width": relation_cross_family_width,
            "max_relation_to_operator_ratio": max_relation_to_operator_ratio,
        }
        write_run_params(run_dir, run_id=resolved_run_id, text=text, config=config)
        write_run_metrics(run_dir, metrics)
        print(f"\nRun directory: {run_dir}")

    total_surface_path = (run_dir / "total_surface.png") if run_dir is not None else None
    total_heatmap_path = (run_dir / "total_heatmap.png") if run_dir is not None else None
    operator_heatmap_path = (run_dir / "operator_heatmap.png") if run_dir is not None else None
    relation_raw_heatmap_path = (run_dir / "relation_raw_heatmap.png") if run_dir is not None else None
    relation_heatmap_path = (run_dir / "relation_normalized_heatmap.png") if run_dir is not None else None

    if show_plots or run_dir is not None:
        plot_surface(
            surface,
            field="total",
            title="Logical Surface of Text v3: Bounded Operators + Relations",
            save_path=total_surface_path,
            show=show_plots,
        )
        plot_heatmap(
            surface,
            field="total",
            title="Total Logical Surface Heatmap v3",
            save_path=total_heatmap_path,
            show=show_plots,
        )
        plot_heatmap(
            surface,
            field="operator",
            title="Operator-Only Logical Field v3",
            save_path=operator_heatmap_path,
            show=show_plots,
        )
        plot_heatmap(
            surface,
            field="relation_raw",
            title="Raw Relation/Scope Logical Field v3",
            save_path=relation_raw_heatmap_path,
            show=show_plots,
        )
        plot_heatmap(
            surface,
            field="relation",
            title="Energy-Normalized Relation/Scope Logical Field v3",
            save_path=relation_heatmap_path,
            show=show_plots,
        )

        if run_dir is not None:
            print("Saved plots:")
            print(f"- {total_surface_path}")
            print(f"- {total_heatmap_path}")
            print(f"- {operator_heatmap_path}")
            print(f"- {relation_raw_heatmap_path}")
            print(f"- {relation_heatmap_path}")

    return {
        "tokens": tokens,
        "boundaries": boundaries,
        "events": events,
        "relations": relations,
        "surface": surface,
        "metrics": metrics,
        "run_id": run_dir.name if run_dir is not None else None,
        "run_dir": run_dir,
    }


def load_text_from_args(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    return """
    If a text contains many conditions, then it may branch into several possible meanings.
    However, if the same text also contains strong causal operators, because it explains
    why one claim follows from another, then the logical flow becomes more structured.
    But when the text repeatedly says maybe, possibly, or not, the surface develops
    uncertainty troughs and contradiction pressure.
    Therefore, this geometric representation can help us compare texts by their
    logical shape rather than only by their words.
    """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logical Surface Model v3")
    parser.add_argument("--text", type=str, default=None, help="Text to analyze directly.")
    parser.add_argument("--file", type=str, default=None, help="Path to a UTF-8 text file to analyze.")
    parser.add_argument("--no-plots", action="store_true", help="Run analysis without showing plots.")
    parser.add_argument("--no-save", action="store_true", help="Skip writing the run directory.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Override the output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Override the auto-generated run id.")
    parser.add_argument("--x-resolution", type=int, default=280, help="Surface resolution along token axis.")
    parser.add_argument("--y-resolution", type=int, default=140, help="Surface resolution along logical-family axis.")
    parser.add_argument(
        "--operator-cross-family-width",
        type=float,
        default=0.55,
        help="Cross-family Gaussian width for operator events.",
    )
    parser.add_argument(
        "--relation-cross-family-width",
        type=float,
        default=0.42,
        help="Cross-family Gaussian width for inferred relations.",
    )
    parser.add_argument(
        "--max-relation-ratio",
        type=float,
        default=1.25,
        help="Maximum allowed normalized relation/operator energy ratio.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    input_text = load_text_from_args(cli_args)
    analyze_text(
        input_text,
        show_plots=not cli_args.no_plots,
        save=not cli_args.no_save,
        output_root=Path(cli_args.output_dir) if cli_args.output_dir else None,
        run_id=cli_args.run_id,
        x_resolution=cli_args.x_resolution,
        y_resolution=cli_args.y_resolution,
        operator_cross_family_width=cli_args.operator_cross_family_width,
        relation_cross_family_width=cli_args.relation_cross_family_width,
        max_relation_to_operator_ratio=cli_args.max_relation_ratio,
    )
