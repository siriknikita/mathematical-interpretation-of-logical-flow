# Experiment Journal — Mathematical Interpretation of Logical Flow

A chronological record of the model's evolution. Each stage documents:

1. **Goal** — what we were trying to express or learn.
2. **Model** — the pipeline and the load-bearing design decisions, with the *why* attached.
3. **Results** — what the model produced on the canonical sample text.
4. **Interpretation** — what the numbers and geometry actually mean.
5. **Limitations & next directions** — what's wrong, missing, or naive, and what the next stage should address.

New stages are appended at the bottom. Older stages are never rewritten — when a decision is revisited, the new stage explains the change against the prior one.

---

## Stage 1 — Toy model (`logical_surface_model_v1.py`)

**Status:** archived as v1. Entry point: `uv run logical_surface_model_v1.py`.

### Goal

Turn an English paragraph into a **3D "logical surface"** — a function `Z(x, y)` where:

- `x` = token position in the text
- `y` = logical family (one of 8 discrete rows)
- `z` = signed logical elevation contributed by detected operators

The aim is **not** to understand the text. The aim is to give the text a *shape* — a geometry — so that later stages can compare texts by their logical contour (ridges, troughs, energy, roughness) rather than by their words.

The guiding intuition is a sign convention:

```
upward waves   = constructive logical flow (cause, condition, conjunction, evidence)
downward waves = negation, opposition, uncertainty, contradiction
branching      = disjunction (small positive but wide)
```

### Model

The pipeline, end to end, lives in a single file:

```
text
 → tokenize                       (regex; keeps contractions and e.g./i.e.)
 → find_operator_events           (longest-pattern-first; non-overlapping)
 → local_context_factor           (±4-token intensifier/hedge modulation)
 → build_surface                  (Mexican-hat × Gaussian on a 240×120 grid)
 → compute_metrics                (mass, energy, roughness, family densities)
 → print + plot_surface + plot_heatmap
```

#### Load-bearing design decisions

1. **`(phrase, family, polarity, strength, width)` as the single source of truth.**
   Every operator is one tuple. The matching index (`OPERATOR_TOKEN_PATTERNS`) and the family-to-Y mapping (`FAMILY_TO_Y`) are derived at import time. **Why:** tuning the model means editing one tuple, not three parallel tables that can drift.

2. **Eight logical families, fixed Y-axis ordering.**
   `conjunction, disjunction, negation, contrast, condition, cause, evidence, uncertainty`. **Why:** the Y axis must be stable across runs and texts so a "contrast ridge near token 40" means the same thing every time. Family order is currently arbitrary but frozen — re-ordering later would invalidate comparisons across previously saved surfaces.

3. **Polarity convention: signed amplitude.**
   `because`, `therefore`, `if`, `then` carry `+polarity`. `not`, `but`, `however`, `unless` carry `-polarity`. Disjunctions are slightly positive but spread out. Hedges (`maybe`, `might`) are mildly negative. **Why:** this is the contract that makes ridges *mean* constructive flow and troughs *mean* destabilization. Every downstream metric (positive/negative mass, net elevation) inherits its meaning from this sign convention.

4. **Tokenization is a regex, not an NLP library.**
   `re.findall(r"e\.g\.|i\.e\.|[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?", text.lower())` — keeps contractions (`can't`) and the two abbreviations we use as operators (`e.g.`, `i.e.`). **Why:** v1 is a geometry experiment, not a parsing experiment. Dependencies stay at `numpy` + `matplotlib`. We accept that this is brittle and revisit it when meaning starts to matter.

5. **Multi-word operators match as token sequences, not substrings.**
   `phrase_to_tokens("as well as")` is `('as', 'well', 'as')`. We never substring-match against the raw text. **Why:** substring matching produces false positives (`but` inside `butler`) and is hard to align with token positions.

6. **Longest-pattern-first matching with an `occupied` mask.**
   `OPERATOR_TOKEN_PATTERNS` is sorted by pattern length descending. Once a span claims tokens, shorter patterns can't reuse them. **Why:** `either or` must beat the standalone `or`; `as a result` must beat the standalone `as`. The order of the catalog tuple no longer matters — the sort handles it.

7. **Local context modulates amplitude, never sign.**
   `local_context_factor` looks ±4 tokens around the match: +0.15 per intensifier (`very`, `really`, `extremely`, …), −0.10 per hedge (`maybe`, `might`, …), clamped to `[0.55, 1.75]`. **Why:** the same operator should be louder in "*very* clearly, therefore X" and softer in "*maybe* therefore X". Clamping prevents pathological texts (a wall of intensifiers) from dominating the surface. Sign stays with the operator — context can't flip meaning.

8. **Mexican-hat (Ricker) wavelet along X, Gaussian along Y.**
   `Z += amplitude · (1 − u²) · exp(−u²/2) · exp(−(dy/0.55)²)` with `u = dx / width`. **Why:** a Gaussian-only kernel gives smooth bumps that say nothing about the *structure* around an operator. A Ricker wavelet gives a peak with flanking troughs, which is the geometry we want — `therefore` *raises* its center but *carves* the immediately adjacent region, modeling the "this concludes a chunk, the next chunk is new ground" intuition. The cross-family Gaussian (`σ = 0.55`) bleeds each event modestly into neighbouring rows so adjacent families couple visibly without smearing globally.

9. **Frozen dataclasses, surface accumulated by addition.**
   `LogicalOperator` and `OperatorEvent` are `frozen=True`. The surface is built by `Z += event_contribution` over all events; there is no interaction term between events at this stage. **Why:** v1 is the linear superposition baseline. If v2 adds relational fields (`if … then …` bridges, `because A, B` arcs), the diff against v1 is exactly that non-linear term.

#### Metrics (the numerical handles)

`compute_metrics` summarizes the surface so we can compare runs without staring at plots:

| Metric                            | Definition                                                         | What it captures                                  |
| --------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------- |
| `operator_density_per_100_tokens` | `len(events) · 100 / token_count`                                  | how operator-laden the prose is                   |
| `positive_wave_mass`              | `sum(Z[Z>0])`                                                      | total constructive elevation (grid-dependent)     |
| `negative_wave_mass`              | `sum(|Z[Z<0]|)`                                                    | total destabilizing depression (grid-dependent)   |
| `net_logical_elevation`           | `mean(Z)`                                                          | whether constructive or destabilizing forces win  |
| `surface_energy`                  | `mean(Z²)`                                                         | how logically intense the text is                 |
| `surface_roughness`               | `mean(sqrt(∂Z/∂x² + ∂Z/∂y²))`                                      | how turbulent the structure is                    |
| `branching_density`               | per-100-tokens count of `disjunction` events                       | how many forks                                    |
| `contrast_pressure`               | per-100-tokens count of `contrast` events                          | how much the text turns against itself            |
| `causal_inference_density`        | per-100-tokens count of (`cause` + `condition`)                    | how much the text derives / explains              |
| `uncertainty_density`             | per-100-tokens count of `uncertainty` events                       | how much the text hedges                          |

### Results — canonical sample text

The script's `__main__` runs on a deliberately operator-dense paragraph (77 tokens, ~1 operator every 5 tokens). Saved plots: `outputs/Figure_1.png` (3D surface), `outputs/Figure_2.png` (heatmap).

#### Detected events (15)

| phrase     | family       | tokens  | amplitude | context |
| ---------- | ------------ | ------- | --------: | ------: |
| if         | condition    | 0:1     |    +0.800 |   1.000 |
| then       | condition    | 6:7     |    +0.630 |   1.000 |
| however    | contrast     | 14:15   |    −0.825 |   1.000 |
| if         | condition    | 15:16   |    +0.800 |   1.000 |
| also       | conjunction  | 19:20   |    +0.150 |   1.000 |
| because    | cause        | 24:25   |    +1.045 |   1.000 |
| then       | condition    | 33:34   |    +0.630 |   1.000 |
| but        | contrast     | 40:41   |    −0.700 |   1.000 |
| when       | condition    | 41:42   |    +0.413 |   1.000 |
| maybe      | uncertainty  | 46:47   |    −0.130 |   0.800 |
| possibly   | uncertainty  | 47:48   |    −0.130 |   0.800 |
| or         | disjunction  | 48:49   |    +0.068 |   0.800 |
| not        | negation     | 49:50   |    −0.800 |   0.800 |
| and        | conjunction  | 55:56   |    +0.193 |   1.000 |
| therefore  | cause        | 58:59   |    +1.150 |   1.000 |

Context factors of `0.800` around tokens 46–50 are the hedge dampening working as designed: `maybe`/`possibly`/`might`/`could`/`seems` are simultaneously *operators* (`uncertainty` family) and *context hedges*, so when they cluster they soften their own neighbours.

#### Surface metrics

```
token_count                      77
operator_count                   15
operator_density_per_100_tokens  19.480519
positive_wave_mass               2407.895070
negative_wave_mass               2246.195479
net_logical_elevation            0.005615
surface_energy                   0.064502
surface_roughness                0.021981
branching_density                1.298701
contrast_pressure                2.597403
causal_inference_density         9.090909
uncertainty_density              2.597403
```

### Interpretation

- The strongest **positive** structures are `therefore (+1.150)`, `because (+1.045)`, `if (+0.800)` — the model correctly reads a **conditional–causal backbone**.
- The strongest **negative** structures are `however (−0.825)`, `not (−0.800)`, `but (−0.700)` — and the model correctly reads **opposition / negation pressure**.
- `positive_wave_mass ≈ 2408` vs. `negative_wave_mass ≈ 2246` gives a balance ratio of about **1.07**. The paragraph is not flat — it has high `surface_energy` — but its constructive and destabilizing forces are *near-balanced*. The plots match: visible ridges and troughs, no global dominance.
- `causal_inference_density ≈ 9.09` is the dominant family signal — about one cause/condition operator per ten tokens. The text reads as an argument scaffold.

Heatmap regions worth noting (by eye, on `outputs/Figure_2.png`):

```
condition ridge      tokens 0–10
contrast trough      tokens 14–18
cause ridge          tokens 22–28
condition ridge      tokens 30–35
contrast/condition   tokens 40–42  (collision: but ↔ when)
neg/uncertainty      tokens 46–50  (maybe possibly or not — a sharp trough)
cause ridge          tokens 58–62  (therefore)
```

That is a usable **logical fingerprint** of the paragraph.

### Limitations of v1 (and what they motivate)

1. **Operators are isolated bumps; relations are missing.**
   `if … then …` produces two separate ridges, not a *bridge* between condition and consequence. `because A, B` does not produce an *arc* connecting cause-span to claim-span. The geometry is local; the logic is not. → **v2 should add a relational field** layered on top of the operator field.

2. **`positive_wave_mass` and `negative_wave_mass` are grid-dependent.**
   Both scale with `x_resolution × y_resolution`. They cannot be compared across surfaces built at different resolutions. → **v2 should report `positive_wave_density = positive_mass / Z.size`** (and the same for negative), plus a `logical_balance_ratio = positive_mass / max(negative_mass, ε)`. The density values are scale-invariant; the ratio is the human-readable summary.

3. **Sentence and clause structure is invisible.**
   The tokenizer flattens punctuation. We don't know whether two operators sit in the same clause or in different sentences. Many of the most interesting comparisons (a `but` *inside* a `because`-clause vs. a `but` between two `because`-clauses) require this. → **v2 should at minimum carry sentence boundaries.**

4. **No notion of operator scope.**
   `not` currently lowers a small fixed region around itself, regardless of whether it negates one word, one clause, or one sentence. Same for `because`. → Scope detection is the prerequisite for relational arcs and for sharper amplitude allocation.

5. **The operator catalog is hand-tuned and English-only.**
   Polarities, strengths, and widths were picked to make the canonical sample text look sensible. They have no empirical basis yet. → Eventually these should be calibrated against a corpus, not author intuition.

6. **Cosmetic: matplotlib warns about tight-layout.**
   Harmless but noisy. → Replace `plt.tight_layout()` with explicit `subplots_adjust` calls in the next stage.

### Candidate next metrics (deferred to v2)

```
logical_tension          — local collisions of +cause/condition with −contrast/negation
argumentative_depth      — nested condition/cause/conclusion relations
branching_complexity     — weighted count of or / if / unless / maybe / could / might
conclusion_force         — culmination strength at therefore / thus / hence / as a result
semantic_stability       — smoothness vs. reversals and uncertainty pockets
```

These all require **relations between scopes**, not just operator events. That's the v2 boundary.

---
