# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A single-file Python experiment that turns English text into a 3D "logical surface." It detects logical operators (conjunction, disjunction, negation, contrast, condition, cause, evidence, uncertainty), places each detection on a token-position × logical-family grid, and sums Mexican-hat (Ricker) wavelets to produce a surface whose ridges and troughs encode the text's logical shape. Reports per-event details, aggregate metrics, and matplotlib 3D + heatmap plots.

Python ≥ 3.13. Managed with `uv` (see `uv.lock`). Dependencies: `numpy`, `matplotlib`.

## Commands

```bash
uv sync             # install/refresh the env from uv.lock
uv run main.py      # run the demo on the embedded sample_text (opens 2 matplotlib windows)
uv add <pkg>        # add a dependency (do not edit pyproject.toml by hand)
```

No tests, linter, or formatter are configured. The script blocks on `plt.show()` until the figure windows are closed — there is no headless / non-interactive mode flag; pass `show_plots=False` to `analyze_text` from a REPL/import if you need silent runs.

## Architecture

Pipeline (all in `main.py`):

1. **Operator catalog** — `LOGICAL_OPERATORS` (`main.py:35`) is the source of truth. Each entry is a `(phrase, family, polarity, strength, width)` tuple. `polarity` ∈ roughly [-1, +1] is the sign/strength of logical "elevation"; `strength` is a per-operator amplitude multiplier; `width` is the spatial extent of its wave along the token axis. To add or retune an operator, edit this tuple — derived structures (`OPERATOR_TOKEN_PATTERNS`, `FAMILY_TO_Y`) rebuild from it at import time.

2. **Tokenization** — `tokenize` (`main.py:112`) is a regex tokenizer that keeps contractions (`can't`) and the `e.g.`/`i.e.` abbreviations. Multi-word operators (`as well as`, `on the other hand`) are matched as **token sequences**, not substrings — `phrase_to_tokens` tokenizes the catalog phrase the same way.

3. **Matching** — `find_operator_events` (`main.py:149`) walks operators **longest-pattern-first** (see the sort at `main.py:123`) so `either or` and `as a result` win over their single-token prefixes. A boolean `occupied` mask prevents overlapping double-counting; once a span is claimed, shorter patterns can't reuse those tokens.

4. **Context modulation** — `local_context_factor` (`main.py:132`) looks ±4 tokens around each match and multiplies amplitude by intensifier presence (`very`, `really`, …) or dampens it by hedge presence (`maybe`, `might`, …). Clamped to [0.55, 1.75]. This is how the same operator phrase produces different amplitudes in different contexts.

5. **Surface construction** — `build_surface` (`main.py:199`) builds a 2D grid `X` (token positions, default 240 samples) × `Y` (8 logical families, default 120 samples) and accumulates `amplitude × mexican_hat_wave(dx, width) × gaussian(dy, cross_family_width)`. The Mexican-hat (Ricker) wavelet at `main.py:188` is what produces ridges-with-flanking-troughs rather than smooth Gaussians; the `cross_family_width=0.55` Gaussian bleeds each event modestly into adjacent family rows. Y-axis ordering is fixed by `FAMILY_ORDER` (`main.py:95`).

6. **Metrics** — `compute_metrics` (`main.py:234`) summarizes the surface: positive/negative mass, mean elevation, mean Z² (surface energy), gradient-magnitude mean (roughness), and per-100-token family densities. These are the numerical handles for comparing texts.

7. **Reporting** — `analyze_text` (`main.py:354`) is the entry point used by `if __name__ == "__main__"`. It prints events + metrics tables and (by default) shows both the 3D surface plot and the 2D heatmap.

## Conventions specific to this code

- Dataclasses are `frozen=True` — events and operators are immutable; don't mutate, construct new ones.
- The longest-first match order in `OPERATOR_TOKEN_PATTERNS` is load-bearing. If you add a multi-word operator whose prefix is also a single-word operator, the sort already handles it — just append to `LOGICAL_OPERATORS`.
- Adding a new **family** requires both a new entry in `LOGICAL_OPERATORS` and appending the family name to `FAMILY_ORDER` (`main.py:95`); the Y-axis labels and `compute_metrics` densities follow from that tuple.
- `outputs/` holds saved example renders (`Figure_1.png`, `Figure_2.png`) — not produced by the script automatically; they were saved by hand from the matplotlib viewer.
