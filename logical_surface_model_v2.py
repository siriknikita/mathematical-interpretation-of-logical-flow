from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# =============================================================================
# Logical Surface Model v2
# =============================================================================
#
# Core idea:
#     Text is treated as a flat symbolic sequence.
#     Logical operators disturb that flat sheet as waves.
#     Operator scopes and relations create longer bridge-fields.
#
# Compared with v1, this version adds:
#     1. sentence-aware tokenization,
#     2. logical operator events,
#     3. heuristic logical scopes,
#     4. relations such as implication, causality, contrast, branching,
#        negation scope, uncertainty scope, and conclusion,
#     5. separate operator/relation/total fields,
#     6. resolution-stable normalized metrics.
#
# Dependencies:
#     pip install numpy matplotlib
#
# Run:
#     python logical_surface_model_v2.py
#     python logical_surface_model_v2.py --text "If A then B, but maybe not C."
#     python logical_surface_model_v2.py --file input.txt --save-prefix output/logical_surface
# =============================================================================


@dataclass(frozen=True)
class Token:
    text: str
    index: int
    char_start: int
    char_end: int
    sentence_index: int


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
    width: float
    sentence_index: int
    label: str


@dataclass(frozen=True)
class SurfaceBundle:
    X: np.ndarray
    Y: np.ndarray
    operator_Z: np.ndarray
    relation_Z: np.ndarray
    total_Z: np.ndarray
    positive_component: np.ndarray
    negative_component: np.ndarray


LOGICAL_OPERATORS: Tuple[LogicalOperator, ...] = (
    # Accumulation / continuation
    LogicalOperator("and", "conjunction", +0.35, 0.55, 4.0),
    LogicalOperator("also", "conjunction", +0.30, 0.50, 4.0),
    LogicalOperator("moreover", "conjunction", +0.45, 0.75, 5.0),
    LogicalOperator("furthermore", "conjunction", +0.45, 0.75, 5.0),
    LogicalOperator("as well as", "conjunction", +0.40, 0.65, 5.0),

    # Branching
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


def tokenize_with_spans(text: str) -> List[Token]:
    """
    Tokenize text and assign a sentence index to every word-like token.

    Punctuation is not returned as a token, but .!?; increment the sentence index.
    Commas are treated as clause hints by the tokenizer but not as sentence breaks.
    """
    pattern = r"e\.g\.|i\.e\.|[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?|[.!?;:,]"
    tokens: List[Token] = []
    sentence_index = 0

    for match in re.finditer(pattern, text.lower()):
        raw = match.group(0)

        if raw in {".", "!", "?", ";"}:
            if tokens:
                sentence_index += 1
            continue

        if raw == ",":
            continue

        tokens.append(
            Token(
                text=raw,
                index=len(tokens),
                char_start=match.start(),
                char_end=match.end(),
                sentence_index=sentence_index,
            )
        )

    return tokens


def phrase_to_tokens(phrase: str) -> Tuple[str, ...]:
    return tuple(token.text for token in tokenize_with_spans(phrase))


OPERATOR_TOKEN_PATTERNS: Tuple[Tuple[Tuple[str, ...], LogicalOperator], ...] = tuple(
    sorted(
        ((phrase_to_tokens(op.phrase), op) for op in LOGICAL_OPERATORS),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def token_texts(tokens: List[Token]) -> List[str]:
    return [token.text for token in tokens]


def sentence_bounds(tokens: List[Token], sentence_index: int) -> Tuple[int, int]:
    indices = [t.index for t in tokens if t.sentence_index == sentence_index]
    if not indices:
        return 0, 0
    return min(indices), max(indices) + 1


def all_sentence_indices(tokens: List[Token]) -> List[int]:
    return sorted(set(t.sentence_index for t in tokens))


def previous_sentence_bounds(tokens: List[Token], sentence_index: int) -> Optional[Tuple[int, int]]:
    previous_indices = [idx for idx in all_sentence_indices(tokens) if idx < sentence_index]
    if not previous_indices:
        return None
    return sentence_bounds(tokens, previous_indices[-1])


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


def local_context_factor(tokens: List[Token], start: int, end: int, window: int = 4) -> float:
    """
    A small heuristic for how strongly the local text amplifies or softens an operator.

    Intensifiers near an operator make the wave taller.
    Hedges near an operator make it softer.
    """
    texts = token_texts(tokens)
    left = max(0, start - window)
    right = min(len(texts), end + window)
    ctx = texts[left:right]

    factor = 1.0
    factor += 0.15 * sum(1 for t in ctx if t in INTENSIFIERS)
    factor -= 0.10 * sum(1 for t in ctx if t in HEDGES)
    return float(np.clip(factor, 0.55, 1.75))


def find_operator_events(text: str) -> Tuple[List[Token], List[OperatorEvent]]:
    tokens = tokenize_with_spans(text)
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
                    )
                )
                occupied[i:i + n] = True

    events.sort(key=lambda e: (e.token_start, e.token_end))
    return tokens, events


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


def make_relation(
    *,
    kind: str,
    family: str,
    source_start: int,
    source_end: int,
    target_start: int,
    target_end: int,
    operator_center: float,
    amplitude: float,
    width: float,
    sentence_index: int,
    label: str,
    token_count: int,
) -> Optional[LogicalRelation]:
    source_start, source_end = clamp_span(source_start, source_end, token_count)
    target_start, target_end = clamp_span(target_start, target_end, token_count)

    if not is_valid_span(source_start, source_end):
        return None

    if not is_valid_span(target_start, target_end):
        target_start, target_end = source_start, source_end

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
        width=width,
        sentence_index=sentence_index,
        label=label,
    )


def infer_logical_relations(tokens: List[Token], events: List[OperatorEvent]) -> List[LogicalRelation]:
    """
    Convert isolated operator events into approximate logical relations.

    This is intentionally heuristic. It is a bridge between pure keyword waves and
    a future dependency-parser/semantic-parser version.
    """
    relations: List[LogicalRelation] = []
    token_count = len(tokens)

    for event in events:
        sent_start, sent_end = sentence_bounds(tokens, event.sentence_index)

        if event.phrase == "if":
            then_event = find_next_event(
                events,
                after_token=event.token_end,
                sentence_index=event.sentence_index,
                phrase="then",
            )

            if then_event is not None:
                relation = make_relation(
                    kind="implication",
                    family="condition",
                    source_start=event.token_end,
                    source_end=then_event.token_start,
                    target_start=then_event.token_end,
                    target_end=sent_end,
                    operator_center=(event.center + then_event.center) / 2.0,
                    amplitude=+0.85 * event.context_factor,
                    width=7.0,
                    sentence_index=event.sentence_index,
                    label="if_scope -> then_scope",
                    token_count=token_count,
                )
            else:
                relation = make_relation(
                    kind="conditional_scope",
                    family="condition",
                    source_start=event.token_end,
                    source_end=sent_end,
                    target_start=event.token_end,
                    target_end=sent_end,
                    operator_center=event.center,
                    amplitude=+0.35 * event.context_factor,
                    width=6.0,
                    sentence_index=event.sentence_index,
                    label="if_scope",
                    token_count=token_count,
                )

            if relation is not None:
                relations.append(relation)

        elif event.phrase in CAUSAL_PHRASES:
            if event.token_start > sent_start:
                source_start = event.token_end
                source_end = sent_end
                target_start = sent_start
                target_end = event.token_start
            else:
                source_start = event.token_end
                source_end = min(sent_end, event.token_end + 10)
                target_start = source_end
                target_end = sent_end

            relation = make_relation(
                kind="causal_support",
                family="cause",
                source_start=source_start,
                source_end=source_end,
                target_start=target_start,
                target_end=target_end,
                operator_center=event.center,
                amplitude=+0.80 * event.context_factor,
                width=7.0,
                sentence_index=event.sentence_index,
                label=f"{event.phrase}_cause -> effect",
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in CONCLUSION_PHRASES:
            if event.token_start > sent_start:
                source_start = sent_start
                source_end = event.token_start
            else:
                previous = previous_sentence_bounds(tokens, event.sentence_index)
                if previous is not None:
                    source_start, source_end = previous
                else:
                    source_start, source_end = sent_start, event.token_start

            relation = make_relation(
                kind="conclusion",
                family="cause",
                source_start=source_start,
                source_end=source_end,
                target_start=event.token_end,
                target_end=sent_end,
                operator_center=event.center,
                amplitude=+0.95 * event.context_factor,
                width=8.0,
                sentence_index=event.sentence_index,
                label=f"premises -> {event.phrase}_conclusion",
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in CONTRAST_PHRASES:
            if event.token_start > sent_start:
                source_start = sent_start
                source_end = event.token_start
            else:
                previous = previous_sentence_bounds(tokens, event.sentence_index)
                if previous is not None:
                    source_start, source_end = previous
                else:
                    source_start, source_end = sent_start, event.token_start

            relation = make_relation(
                kind="contrast",
                family="contrast",
                source_start=source_start,
                source_end=source_end,
                target_start=event.token_end,
                target_end=sent_end,
                operator_center=event.center,
                amplitude=-0.75 * event.context_factor,
                width=8.0,
                sentence_index=event.sentence_index,
                label=f"left_claim <-> {event.phrase}_right_claim",
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in BRANCHING_PHRASES:
            relation = make_relation(
                kind="branching",
                family="disjunction",
                source_start=max(sent_start, event.token_start - 5),
                source_end=event.token_start,
                target_start=event.token_end,
                target_end=min(sent_end, event.token_end + 5),
                operator_center=event.center,
                amplitude=+0.45 * event.context_factor,
                width=5.0,
                sentence_index=event.sentence_index,
                label=f"branch_left {event.phrase} branch_right",
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in NEGATION_PHRASES or event.phrase == "unless":
            scope_end = min(sent_end, event.token_end + 8)
            relation = make_relation(
                kind="negation_scope",
                family="negation",
                source_start=event.token_start,
                source_end=scope_end,
                target_start=event.token_start,
                target_end=scope_end,
                operator_center=event.center,
                amplitude=-0.60 * event.context_factor,
                width=5.0,
                sentence_index=event.sentence_index,
                label=f"{event.phrase}_scope",
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

        elif event.phrase in UNCERTAINTY_PHRASES:
            scope_end = min(sent_end, event.token_end + 8)
            relation = make_relation(
                kind="uncertainty_scope",
                family="uncertainty",
                source_start=event.token_start,
                source_end=scope_end,
                target_start=event.token_start,
                target_end=scope_end,
                operator_center=event.center,
                amplitude=-0.35 * event.context_factor,
                width=6.0,
                sentence_index=event.sentence_index,
                label=f"{event.phrase}_scope",
                token_count=token_count,
            )
            if relation is not None:
                relations.append(relation)

    relations.sort(key=lambda relation: (relation.source_start, relation.target_start, relation.kind))
    return relations


def mexican_hat_wave(distance: np.ndarray, width: float) -> np.ndarray:
    """
    Ricker / Mexican-hat style wave:
    positive or negative center with surrounding correction.

    This produces ridges and troughs rather than simple Gaussian bumps.
    """
    safe_width = max(width, 1e-6)
    x = distance / safe_width
    return (1.0 - x ** 2) * np.exp(-0.5 * x ** 2)


def gaussian_kernel(distance: np.ndarray, width: float) -> np.ndarray:
    safe_width = max(width, 1e-6)
    return np.exp(-0.5 * (distance / safe_width) ** 2)


def interval_kernel(X: np.ndarray, start: float, end: float, edge_width: float) -> np.ndarray:
    """
    Smoothly activate a field over an interval on the token-position axis.

    Inside the interval the kernel is close to 1.
    Outside the interval it decays as a Gaussian.
    """
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


def add_component(
    field: np.ndarray,
    total: np.ndarray,
    positive_component: np.ndarray,
    negative_component: np.ndarray,
) -> None:
    total += field
    positive_component += np.clip(field, 0.0, None)
    negative_component += np.clip(-field, 0.0, None)


def build_surface(
    tokens: List[Token],
    events: List[OperatorEvent],
    relations: List[LogicalRelation],
    x_resolution: int = 280,
    y_resolution: int = 140,
    operator_cross_family_width: float = 0.55,
    relation_cross_family_width: float = 0.75,
) -> SurfaceBundle:
    """
    Creates fields over:
        X = token-position axis
        Y = logical-family axis
        Z = logical elevation

    operator_Z:
        local waves produced by detected logical operators.

    relation_Z:
        longer fields produced by inferred logical scopes and relations.

    total_Z:
        operator_Z + relation_Z.
    """
    if len(tokens) == 0:
        x = np.linspace(0, 1, x_resolution)
    else:
        x = np.linspace(0, max(1, len(tokens) - 1), x_resolution)

    y = np.linspace(0, len(FAMILY_ORDER) - 1, y_resolution)
    X, Y = np.meshgrid(x, y)

    operator_Z = np.zeros_like(X, dtype=float)
    relation_Z = np.zeros_like(X, dtype=float)
    positive_component = np.zeros_like(X, dtype=float)
    negative_component = np.zeros_like(X, dtype=float)

    for event in events:
        family_y = FAMILY_TO_Y[event.family]
        dx = X - event.center
        dy = Y - family_y

        longitudinal_wave = mexican_hat_wave(dx, event.width)
        family_kernel = gaussian_kernel(dy, operator_cross_family_width)
        field = event.amplitude * longitudinal_wave * family_kernel

        add_component(field, operator_Z, positive_component, negative_component)

    for relation in relations:
        family_y = FAMILY_TO_Y[relation.family]

        start = min(
            relation.source_start,
            relation.source_end - 1,
            relation.target_start,
            relation.target_end - 1,
            relation.operator_center,
        )
        end = max(
            relation.source_start,
            relation.source_end - 1,
            relation.target_start,
            relation.target_end - 1,
            relation.operator_center,
        )

        dx_kernel = interval_kernel(X, start, end, edge_width=relation.width)
        dy_kernel = gaussian_kernel(Y - family_y, relation_cross_family_width)

        if relation.kind == "branching":
            length = max(end - start, 1e-6)
            phase = np.pi * (X - start) / length
            bridge_shape = np.sin(np.clip(phase, 0.0, np.pi))
            field = relation.amplitude * dx_kernel * bridge_shape * dy_kernel
        elif relation.kind in {"negation_scope", "uncertainty_scope", "conditional_scope"}:
            scope_start = relation.source_start
            scope_end = relation.source_end - 1
            scope_kernel = interval_kernel(X, scope_start, scope_end, edge_width=relation.width)
            field = relation.amplitude * scope_kernel * dy_kernel
        else:
            center = (start + end) / 2.0
            length = max(end - start, 1e-6)
            mild_arch = 0.75 + 0.25 * np.cos(np.pi * (X - center) / max(length, 1e-6))
            mild_arch = np.clip(mild_arch, 0.50, 1.00)
            field = relation.amplitude * dx_kernel * mild_arch * dy_kernel

        add_component(field, relation_Z, positive_component, negative_component)

    total_Z = operator_Z + relation_Z

    return SurfaceBundle(
        X=X,
        Y=Y,
        operator_Z=operator_Z,
        relation_Z=relation_Z,
        total_Z=total_Z,
        positive_component=positive_component,
        negative_component=negative_component,
    )


def compute_max_relation_overlap(relations: List[LogicalRelation]) -> int:
    """
    Approximate argumentative depth by counting maximum overlapping relation spans.
    """
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


def compute_scope_coverage(token_count: int, relations: List[LogicalRelation]) -> float:
    if token_count <= 0 or not relations:
        return 0.0

    covered = np.zeros(token_count, dtype=bool)

    for relation in relations:
        source_start, source_end = clamp_span(relation.source_start, relation.source_end, token_count)
        target_start, target_end = clamp_span(relation.target_start, relation.target_end, token_count)
        covered[source_start:source_end] = True
        covered[target_start:target_end] = True

    return float(np.mean(covered))


def compute_metrics(
    tokens: List[Token],
    events: List[OperatorEvent],
    relations: List[LogicalRelation],
    surface: SurfaceBundle,
) -> Dict[str, float]:
    token_count = len(tokens)
    sentence_count = len(all_sentence_indices(tokens))
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

    semantic_stability = 1.0 / (
        1.0
        + roughness
        + tension
        + negative_density
        + 0.10 * contrast_relation_pressure
        + 0.10 * uncertainty_scope_pressure
    )

    metrics = {
        "token_count": float(token_count),
        "sentence_count": float(sentence_count),
        "operator_count": float(len(events)),
        "relation_count": float(len(relations)),
        "operator_density_per_100_tokens": float(len(events) * per_100),
        "relation_density_per_100_tokens": float(len(relations) * per_100),

        "positive_wave_density": positive_density,
        "negative_wave_density": negative_density,
        "logical_balance_ratio": float(positive_density / max(negative_density, 1e-12)),
        "net_logical_elevation": float(np.mean(Z)),
        "surface_energy_density": float(np.mean(Z ** 2)),
        "surface_roughness_density": roughness,
        "logical_tension_density": tension,

        "operator_field_energy_density": float(np.mean(surface.operator_Z ** 2)),
        "relation_field_energy_density": float(np.mean(surface.relation_Z ** 2)),
        "relation_to_operator_energy_ratio": float(
            np.mean(surface.relation_Z ** 2) / max(np.mean(surface.operator_Z ** 2), 1e-12)
        ),

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
        "negation_scope_density": float(relation_counts.get("negation_scope", 0) * per_100),
        "uncertainty_scope_density": float(relation_counts.get("uncertainty_scope", 0) * per_100),

        "conclusion_force": float(conclusion_force),
        "contrast_relation_pressure": float(contrast_relation_pressure),
        "negation_scope_pressure": float(negation_scope_pressure),
        "uncertainty_scope_pressure": float(uncertainty_scope_pressure),
        "argumentative_depth": float(compute_max_relation_overlap(relations)),
        "scope_coverage_ratio": float(compute_scope_coverage(token_count, relations)),
        "semantic_stability": float(semantic_stability),
    }

    return metrics


def format_span(tokens: List[Token], start: int, end: int, max_tokens: int = 10) -> str:
    start, end = clamp_span(start, end, len(tokens))
    texts = [token.text for token in tokens[start:end]]

    if len(texts) > max_tokens:
        texts = texts[:max_tokens] + ["..."]

    return " ".join(texts)


def print_events(events: List[OperatorEvent]) -> None:
    if not events:
        print("No logical operators were detected.")
        return

    print("\nDetected logical-operator events:")
    print("-" * 112)
    print(
        f"{'phrase':<22} {'family':<14} {'sentence':>8} {'tokens':<12} "
        f"{'amplitude':>10} {'context':>9}"
    )
    print("-" * 112)

    for event in events:
        token_span = f"{event.token_start}:{event.token_end}"
        print(
            f"{event.phrase:<22} {event.family:<14} {event.sentence_index:>8} "
            f"{token_span:<12} {event.amplitude:>10.3f} {event.context_factor:>9.3f}"
        )


def print_relations(tokens: List[Token], relations: List[LogicalRelation]) -> None:
    if not relations:
        print("\nNo logical relations were inferred.")
        return

    print("\nInferred logical relations:")
    print("-" * 140)
    print(
        f"{'kind':<22} {'family':<12} {'sentence':>8} {'amplitude':>10} "
        f"{'source span':<36} {'target span':<36} {'label':<24}"
    )
    print("-" * 140)

    for relation in relations:
        source = format_span(tokens, relation.source_start, relation.source_end)
        target = format_span(tokens, relation.target_start, relation.target_end)

        print(
            f"{relation.kind:<22} {relation.family:<12} {relation.sentence_index:>8} "
            f"{relation.amplitude:>10.3f} {source:<36} {target:<36} {relation.label:<24}"
        )


def print_metrics(metrics: Dict[str, float]) -> None:
    print("\nSurface/text metrics:")
    print("-" * 112)

    integer_keys = {
        "token_count",
        "sentence_count",
        "operator_count",
        "relation_count",
        "argumentative_depth",
    }

    for key, value in metrics.items():
        if key in integer_keys:
            print(f"{key:<42} {int(value)}")
        else:
            print(f"{key:<42} {value:.6f}")


def interpret_metrics(metrics: Dict[str, float]) -> str:
    """
    Produce a compact human-readable interpretation.
    """
    balance = metrics["logical_balance_ratio"]
    tension = metrics["logical_tension_density"]
    relation_ratio = metrics["relation_to_operator_energy_ratio"]
    stability = metrics["semantic_stability"]

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

    if relation_ratio > 1.00:
        relation_text = "long-range relations dominate over isolated operators"
    elif relation_ratio > 0.35:
        relation_text = "relations substantially shape the surface"
    else:
        relation_text = "the surface is mostly driven by local operators"

    if stability > 0.80:
        stability_text = "high semantic stability"
    elif stability > 0.55:
        stability_text = "moderate semantic stability"
    else:
        stability_text = "low semantic stability"

    return (
        f"Interpretation: {balance_text}; {tension_text}; "
        f"{relation_text}; {stability_text}."
    )


def plot_surface(
    surface: SurfaceBundle,
    field: str = "total",
    title: str = "Logical Surface of Text",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    if field == "operator":
        Z = surface.operator_Z
    elif field == "relation":
        Z = surface.relation_Z
    elif field == "total":
        Z = surface.total_Z
    else:
        raise ValueError("field must be one of: operator, relation, total")

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
    if field == "operator":
        Z = surface.operator_Z
    elif field == "relation":
        Z = surface.relation_Z
    elif field == "total":
        Z = surface.total_Z
    else:
        raise ValueError("field must be one of: operator, relation, total")

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


def analyze_text(
    text: str,
    *,
    show_plots: bool = True,
    save_prefix: Optional[str] = None,
    x_resolution: int = 280,
    y_resolution: int = 140,
) -> Dict[str, object]:
    tokens, events = find_operator_events(text)
    relations = infer_logical_relations(tokens, events)
    surface = build_surface(
        tokens,
        events,
        relations,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
    )
    metrics = compute_metrics(tokens, events, relations, surface)

    print(f"Tokenized text into {len(tokens)} tokens across {len(all_sentence_indices(tokens))} sentence block(s).")
    print_events(events)
    print_relations(tokens, relations)
    print_metrics(metrics)
    print("\n" + interpret_metrics(metrics))

    prefix_path: Optional[Path] = None
    if save_prefix is not None:
        prefix_path = Path(save_prefix)
        if prefix_path.parent != Path("."):
            prefix_path.parent.mkdir(parents=True, exist_ok=True)

    if show_plots or prefix_path is not None:
        total_surface_path = None if prefix_path is None else prefix_path.with_name(prefix_path.name + "_total_surface.png")
        total_heatmap_path = None if prefix_path is None else prefix_path.with_name(prefix_path.name + "_total_heatmap.png")
        operator_heatmap_path = None if prefix_path is None else prefix_path.with_name(prefix_path.name + "_operator_heatmap.png")
        relation_heatmap_path = None if prefix_path is None else prefix_path.with_name(prefix_path.name + "_relation_heatmap.png")

        plot_surface(
            surface,
            field="total",
            title="Logical Surface of Text v2: Operators + Relations",
            save_path=total_surface_path,
            show=show_plots,
        )
        plot_heatmap(
            surface,
            field="total",
            title="Total Logical Surface Heatmap",
            save_path=total_heatmap_path,
            show=show_plots,
        )
        plot_heatmap(
            surface,
            field="operator",
            title="Operator-Only Logical Field",
            save_path=operator_heatmap_path,
            show=show_plots,
        )
        plot_heatmap(
            surface,
            field="relation",
            title="Relation/Scope Logical Field",
            save_path=relation_heatmap_path,
            show=show_plots,
        )

        if prefix_path is not None:
            print("\nSaved plots:")
            print(f"- {total_surface_path}")
            print(f"- {total_heatmap_path}")
            print(f"- {operator_heatmap_path}")
            print(f"- {relation_heatmap_path}")

    return {
        "tokens": tokens,
        "events": events,
        "relations": relations,
        "surface": surface,
        "metrics": metrics,
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
    parser = argparse.ArgumentParser(description="Logical Surface Model v2")
    parser.add_argument("--text", type=str, default=None, help="Text to analyze directly.")
    parser.add_argument("--file", type=str, default=None, help="Path to a UTF-8 text file to analyze.")
    parser.add_argument("--no-plots", action="store_true", help="Run analysis without showing plots.")
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Save generated plots using this prefix, for example output/logical_surface.",
    )
    parser.add_argument("--x-resolution", type=int, default=280, help="Surface resolution along token axis.")
    parser.add_argument("--y-resolution", type=int, default=140, help="Surface resolution along logical-family axis.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    input_text = load_text_from_args(cli_args)

    analyze_text(
        input_text,
        show_plots=not cli_args.no_plots,
        save_prefix=cli_args.save_prefix,
        x_resolution=cli_args.x_resolution,
        y_resolution=cli_args.y_resolution,
    )
