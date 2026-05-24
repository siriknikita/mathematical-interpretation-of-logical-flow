
from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


VERSION_TAG = "v1"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


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


LOGICAL_OPERATORS: Tuple[LogicalOperator, ...] = (
    LogicalOperator("and", "conjunction", +0.35, 0.55, 4.0),
    LogicalOperator("also", "conjunction", +0.30, 0.50, 4.0),
    LogicalOperator("moreover", "conjunction", +0.45, 0.75, 5.0),
    LogicalOperator("furthermore", "conjunction", +0.45, 0.75, 5.0),
    LogicalOperator("as well as", "conjunction", +0.40, 0.65, 5.0),

    LogicalOperator("or", "disjunction", +0.10, 0.85, 5.0),
    LogicalOperator("either", "disjunction", +0.10, 0.65, 5.0),
    LogicalOperator("either or", "disjunction", +0.15, 1.00, 6.0),
    LogicalOperator("alternatively", "disjunction", +0.10, 0.90, 6.0),

    LogicalOperator("not", "negation", -1.00, 1.00, 4.0),
    LogicalOperator("no", "negation", -0.90, 0.85, 4.0),
    LogicalOperator("never", "negation", -1.00, 1.10, 5.0),
    LogicalOperator("cannot", "negation", -1.00, 1.00, 4.0),
    LogicalOperator("can't", "negation", -1.00, 1.00, 4.0),
    LogicalOperator("without", "negation", -0.75, 0.80, 5.0),

    LogicalOperator("but", "contrast", -0.70, 1.00, 6.0),
    LogicalOperator("however", "contrast", -0.75, 1.10, 7.0),
    LogicalOperator("although", "contrast", -0.55, 0.90, 7.0),
    LogicalOperator("though", "contrast", -0.45, 0.80, 6.0),
    LogicalOperator("nevertheless", "contrast", -0.80, 1.15, 7.0),
    LogicalOperator("despite", "contrast", -0.60, 0.95, 7.0),
    LogicalOperator("whereas", "contrast", -0.65, 0.95, 7.0),
    LogicalOperator("on the other hand", "contrast", -0.80, 1.20, 8.0),

    LogicalOperator("if", "condition", +0.80, 1.00, 6.0),
    LogicalOperator("then", "condition", +0.70, 0.90, 6.0),
    LogicalOperator("when", "condition", +0.55, 0.75, 6.0),
    LogicalOperator("whenever", "condition", +0.65, 0.85, 6.0),
    LogicalOperator("unless", "condition", -0.85, 1.10, 7.0),
    LogicalOperator("provided that", "condition", +0.75, 1.00, 7.0),
    LogicalOperator("in case", "condition", +0.60, 0.85, 7.0),

    LogicalOperator("because", "cause", +0.95, 1.10, 7.0),
    LogicalOperator("since", "cause", +0.65, 0.85, 6.0),
    LogicalOperator("therefore", "cause", +1.00, 1.15, 7.0),
    LogicalOperator("thus", "cause", +1.00, 1.10, 7.0),
    LogicalOperator("hence", "cause", +1.00, 1.10, 7.0),
    LogicalOperator("so", "cause", +0.75, 0.85, 6.0),
    LogicalOperator("as a result", "cause", +1.00, 1.20, 8.0),
    LogicalOperator("consequently", "cause", +1.00, 1.20, 8.0),
    LogicalOperator("due to", "cause", +0.75, 0.90, 7.0),

    LogicalOperator("for example", "evidence", +0.70, 0.80, 7.0),
    LogicalOperator("for instance", "evidence", +0.70, 0.80, 7.0),
    LogicalOperator("such as", "evidence", +0.60, 0.70, 6.0),
    LogicalOperator("e.g.", "evidence", +0.60, 0.70, 5.0),
    LogicalOperator("namely", "evidence", +0.65, 0.75, 6.0),

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

INTENSIFIERS = {"very", "really", "extremely", "strongly", "clearly", "definitely", "must", "always"}
HEDGES = {"maybe", "possibly", "probably", "somewhat", "kind", "sort", "might", "could", "seems"}


def tokenize(text: str) -> List[str]:
    """
    Lowercase tokenizer that keeps contractions and simple abbreviations.
    """
    return re.findall(r"e\.g\.|i\.e\.|[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?", text.lower())


def phrase_to_tokens(phrase: str) -> Tuple[str, ...]:
    return tuple(tokenize(phrase))


OPERATOR_TOKEN_PATTERNS: Tuple[Tuple[Tuple[str, ...], LogicalOperator], ...] = tuple(
    sorted(
        ((phrase_to_tokens(op.phrase), op) for op in LOGICAL_OPERATORS),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def local_context_factor(tokens: List[str], start: int, end: int, window: int = 4) -> float:
    """
    A small heuristic for how strongly the local text amplifies or softens an operator.

    Intensifiers near an operator make the wave taller.
    Hedges near an operator make it softer.
    """
    left = max(0, start - window)
    right = min(len(tokens), end + window)
    ctx = tokens[left:right]

    factor = 1.0
    factor += 0.15 * sum(1 for t in ctx if t in INTENSIFIERS)
    factor -= 0.10 * sum(1 for t in ctx if t in HEDGES)
    return float(np.clip(factor, 0.55, 1.75))


def find_operator_events(text: str) -> Tuple[List[str], List[OperatorEvent]]:
    tokens = tokenize(text)
    events: List[OperatorEvent] = []

    occupied = np.zeros(len(tokens), dtype=bool)

    for pattern, op in OPERATOR_TOKEN_PATTERNS:
        n = len(pattern)
        if n == 0:
            continue

        for i in range(0, len(tokens) - n + 1):
            if occupied[i:i + n].any():
                continue
            if tuple(tokens[i:i + n]) == pattern:
                center = i + (n - 1) / 2.0
                factor = local_context_factor(tokens, i, i + n)
                amplitude = op.polarity * op.strength * factor

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
                    )
                )
                occupied[i:i + n] = True

    events.sort(key=lambda e: (e.token_start, e.token_end))
    return tokens, events


def mexican_hat_wave(distance: np.ndarray, width: float) -> np.ndarray:
    """
    Ricker / Mexican-hat style wave:
    positive or negative center with surrounding correction.

    This produces surfaces with ridges and troughs rather than boring Gaussian bumps.
    """
    x = distance / width
    return (1.0 - x ** 2) * np.exp(-0.5 * x ** 2)


def build_surface(
    tokens: List[str],
    events: List[OperatorEvent],
    x_resolution: int = 240,
    y_resolution: int = 120,
    cross_family_width: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a 3D surface Z over:
    X = token-position axis
    Y = logical-family axis
    Z = logical wave height
    """
    if len(tokens) == 0:
        x = np.linspace(0, 1, x_resolution)
    else:
        x = np.linspace(0, max(1, len(tokens) - 1), x_resolution)

    y = np.linspace(0, len(FAMILY_ORDER) - 1, y_resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=float)

    for event in events:
        family_y = FAMILY_TO_Y[event.family]
        dx = X - event.center
        dy = Y - family_y

        longitudinal_wave = mexican_hat_wave(dx, event.width)
        family_kernel = np.exp(-0.5 * (dy / cross_family_width) ** 2)

        Z += event.amplitude * longitudinal_wave * family_kernel

    return X, Y, Z


def compute_metrics(tokens: List[str], events: List[OperatorEvent], Z: np.ndarray) -> Dict[str, float]:
    token_count = len(tokens)

    if len(events) == 0:
        return {
            "token_count": float(token_count),
            "operator_count": 0.0,
            "operator_density_per_100_tokens": 0.0,
            "positive_wave_mass": 0.0,
            "negative_wave_mass": 0.0,
            "net_logical_elevation": 0.0,
            "surface_energy": 0.0,
            "surface_roughness": 0.0,
            "branching_density": 0.0,
            "contrast_pressure": 0.0,
            "causal_inference_density": 0.0,
            "uncertainty_density": 0.0,
        }

    positive_mass = float(np.sum(Z[Z > 0]))
    negative_mass = float(np.sum(np.abs(Z[Z < 0])))
    net_elevation = float(np.mean(Z))
    surface_energy = float(np.mean(Z ** 2))

    gy, gx = np.gradient(Z)
    roughness = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    per_100 = 100.0 / max(1, token_count)

    def family_count(family: str) -> int:
        return sum(1 for e in events if e.family == family)

    return {
        "token_count": float(token_count),
        "operator_count": float(len(events)),
        "operator_density_per_100_tokens": float(len(events) * per_100),
        "positive_wave_mass": positive_mass,
        "negative_wave_mass": negative_mass,
        "net_logical_elevation": net_elevation,
        "surface_energy": surface_energy,
        "surface_roughness": roughness,
        "branching_density": float(family_count("disjunction") * per_100),
        "contrast_pressure": float(family_count("contrast") * per_100),
        "causal_inference_density": float((family_count("cause") + family_count("condition")) * per_100),
        "uncertainty_density": float(family_count("uncertainty") * per_100),
    }


def print_events(events: List[OperatorEvent]) -> None:
    if not events:
        print("No logical operators were detected.")
        return

    print("\nDetected logical-operator events:")
    print("-" * 100)
    print(f"{'phrase':<22} {'family':<14} {'tokens':<12} {'amplitude':>10} {'context':>9}")
    print("-" * 100)

    for e in events:
        token_span = f"{e.token_start}:{e.token_end}"
        print(
            f"{e.phrase:<22} {e.family:<14} {token_span:<12} "
            f"{e.amplitude:>10.3f} {e.context_factor:>9.3f}"
        )


def print_metrics(metrics: Dict[str, float]) -> None:
    print("\nSurface/text metrics:")
    print("-" * 100)
    for key, value in metrics.items():
        if key in {"token_count", "operator_count"}:
            print(f"{key:<36} {int(value)}")
        else:
            print(f"{key:<36} {value:.6f}")


def plot_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str = "Logical Surface of Text",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X,
        Y,
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

    fig.colorbar(surface, shrink=0.55, aspect=12, pad=0.08, label="Elevation")
    fig.subplots_adjust(left=0.04, right=0.86, bottom=0.08, top=0.92)

    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_heatmap(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str = "Logical Surface Heatmap",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    plotted = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
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
# Run artifact persistence
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
    x_resolution: int = 240,
    y_resolution: int = 120,
    cross_family_width: float = 0.55,
) -> Dict[str, object]:
    tokens, events = find_operator_events(text)
    X, Y, Z = build_surface(
        tokens,
        events,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        cross_family_width=cross_family_width,
    )
    metrics = compute_metrics(tokens, events, Z)

    print(f"Tokenized text into {len(tokens)} tokens.")
    print_events(events)
    print_metrics(metrics)

    run_dir: Optional[Path] = None
    if save:
        root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
        resolved_run_id = run_id or generate_run_id(text)
        run_dir = prepare_run_directory(root, resolved_run_id)
        config = {
            "x_resolution": x_resolution,
            "y_resolution": y_resolution,
            "cross_family_width": cross_family_width,
        }
        write_run_params(run_dir, run_id=resolved_run_id, text=text, config=config)
        write_run_metrics(run_dir, metrics)
        print(f"\nRun directory: {run_dir}")

    surface_path = (run_dir / "surface.png") if run_dir is not None else None
    heatmap_path = (run_dir / "heatmap.png") if run_dir is not None else None

    if show_plots or run_dir is not None:
        plot_surface(X, Y, Z, save_path=surface_path, show=show_plots)
        plot_heatmap(X, Y, Z, save_path=heatmap_path, show=show_plots)

        if run_dir is not None:
            print("Saved plots:")
            print(f"- {surface_path}")
            print(f"- {heatmap_path}")

    return {
        "tokens": tokens,
        "events": events,
        "X": X,
        "Y": Y,
        "Z": Z,
        "metrics": metrics,
        "run_id": run_id if run_id is not None else (run_dir.name if run_dir is not None else None),
        "run_dir": run_dir,
    }


SAMPLE_TEXT = """
If a text contains many conditions, then it may branch into several possible meanings.
However, if the same text also contains strong causal operators, because it explains
why one claim follows from another, then the logical flow becomes more structured.
But when the text repeatedly says maybe, possibly, or not, the surface develops
uncertainty troughs and contradiction pressure.
Therefore, this geometric representation can help us compare texts by their
logical shape rather than only by their words.
"""


def load_text_from_args(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    return SAMPLE_TEXT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logical Surface Model v1")
    parser.add_argument("--text", type=str, default=None, help="Text to analyze directly.")
    parser.add_argument("--file", type=str, default=None, help="Path to a UTF-8 text file to analyze.")
    parser.add_argument("--no-plots", action="store_true", help="Skip displaying matplotlib windows.")
    parser.add_argument("--no-save", action="store_true", help="Skip writing the run directory.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Override the output root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Override the auto-generated run id.")
    parser.add_argument("--x-resolution", type=int, default=240, help="Surface resolution along token axis.")
    parser.add_argument("--y-resolution", type=int, default=120, help="Surface resolution along logical-family axis.")
    parser.add_argument(
        "--cross-family-width",
        type=float,
        default=0.55,
        help="Gaussian width that bleeds an event into adjacent logical families.",
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
        cross_family_width=cli_args.cross_family_width,
    )
