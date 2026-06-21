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

## Stage 2 — Scope/relation field model (`logical_surface_model_v2.py`)

**Status:** current. Entry point: `uv run logical_surface_model_v2.py` (with `--text`, `--file`, `--no-plots`, `--save-prefix`, `--x-resolution`, `--y-resolution`).

### Goal

Promote the model from *isolated operator bumps* to a model that detects **scopes** for each operator and **relations between scopes**, then builds the surface as the sum of two distinct fields:

```
text
 → tokens (sentence-aware)
 → operator events
 → logical scopes
 → logical relations between scopes
 → operator_Z (local waves)  +  relation_Z (bridge fields)
 → total_Z = operator_Z + relation_Z
 → resolution-stable metrics
```

The v1 surface is preserved as `operator_Z`; the new contribution is `relation_Z`. This makes the v2 result interpretable as **v1 plus a relational layer**, and `relation_to_operator_energy_ratio` becomes the explicit handle for how much of the surface is operator-driven vs. relation-driven.

### Model

#### Diff against v1 (only what changed)

| # | Decision | v1 | v2 | Why |
| - | -------- | -- | -- | --- |
| 1 | Tokenization | flat token list | sentence-aware `Token(text, index, char_start, char_end, sentence_index)`; `.!?;` increment the sentence index; `,` ignored | scope inference needs sentence boundaries to stop spilling across full stops |
| 2 | Operator event | `OperatorEvent` (v1 fields) | adds `sentence_index` | so a relation knows which sentence it lives in |
| 3 | Logical relation | absent | new `LogicalRelation(kind, family, source_span, target_span, operator_center, amplitude, width, sentence_index, label)` | the relation is now a first-class object with both endpoints, an inferred kind, and a directionality label |
| 4 | Surface | one `Z` | `SurfaceBundle` with `operator_Z`, `relation_Z`, `total_Z`, plus `positive_component` and `negative_component` accumulators | lets the two layers be inspected, plotted, and quantified separately |
| 5 | Wave kernels | Ricker × Gaussian only | adds `interval_kernel` (flat-top with Gaussian edges) and a `sin`-shaped bridge for branching | relations are not point events; they need spatially-extended kernels |
| 6 | Metric normalization | `positive_wave_mass` (grid-dependent) | `positive_wave_density = sum(positive) / size`, plus `logical_balance_ratio = pos_density / max(neg_density, ε)` | makes metrics scale-invariant — solves v1's grid-dependence limitation |
| 7 | New metrics | — | `logical_tension_density`, `relation_to_operator_energy_ratio`, `argumentative_depth` (max overlap of relation spans), `scope_coverage_ratio`, `semantic_stability`, per-relation-kind densities, `conclusion_force`, `contrast_relation_pressure`, `negation_scope_pressure`, `uncertainty_scope_pressure` | the metrics promised at the end of Stage 1, now actually computable because we have relations |
| 8 | Reporting | events + metrics | events + **relations table (kind, family, source span, target span, amplitude, label)** + metrics + `interpret_metrics` one-line summary | a human can read the inferred logic without the plots |
| 9 | Plots | one 3D + one heatmap | 3D total surface + 3 heatmaps (total / operator-only / relation-only) | the operator and relation layers must be visually separable for diagnosis |
| 10 | CLI / outputs | none | `argparse` with `--text` / `--file` / `--no-plots` / `--save-prefix` / `--x-resolution` / `--y-resolution`; deterministic file naming under `save-prefix` | makes the model usable on arbitrary texts and reproducible |
| 11 | Cosmetic | `tight_layout` warning | replaced with explicit `subplots_adjust` | the v1 noise is gone |

Everything from v1 that wasn't changed — the operator catalog, `LOGICAL_OPERATORS` tuple, polarity convention, longest-pattern-first matching, the `occupied` mask, the Mexican-hat wavelet for operator events, `local_context_factor` — is **kept verbatim**. v2 is strictly additive on top of v1.

#### Relation inference rules (`infer_logical_relations`, `logical_surface_model_v2.py:445`)

The heart of v2. For each operator event, a relation is synthesized using sentence bounds and the position of the operator inside its sentence. All rules are sentence-local — no relation crosses a `.!?;`. The bridge fields are scoped to `[min(spans, operator), max(spans, operator)]`.

| Operator class | Relation `kind` | Span construction | Amplitude (× context_factor) | Width |
| -------------- | --------------- | ----------------- | ---------------------------: | ----: |
| `if` followed by `then` in same sentence | `implication` | source = (after-`if`, before-`then`), target = (after-`then`, sentence end) | +0.85 | 7.0 |
| `if` without matching `then` | `conditional_scope` | source = target = (after-`if`, sentence end) | +0.35 | 6.0 |
| `because` / `since` / `due to` | `causal_support` | if mid-sentence: source = (after-op, sentence end), target = (sentence start, before-op); else fall-through both sides forward | +0.80 | 7.0 |
| `therefore` / `thus` / `hence` / `so` / `as a result` / `consequently` | `conclusion` | source = (sentence start, before-op) **or** previous sentence if sentence-initial; target = (after-op, sentence end) | +0.95 | 8.0 |
| `but` / `however` / `although` / `nevertheless` / `despite` / `whereas` / `on the other hand` | `contrast` | source = (sentence start, before-op) or previous sentence if sentence-initial; target = (after-op, sentence end) | −0.75 | 8.0 |
| `or` / `either` / `either or` / `alternatively` | `branching` | source = (op−5, op), target = (op, op+5); shaped by `sin(π · (x−start)/length)` | +0.45 | 5.0 |
| `not` / `no` / `never` / `cannot` / `can't` / `without` / `unless` | `negation_scope` | source = target = (op, op+8) clamped to sentence | −0.60 | 5.0 |
| `maybe` / `possibly` / `probably` / `might` / `could` / `seems` | `uncertainty_scope` | source = target = (op, op+8) clamped to sentence | −0.35 | 6.0 |

Three field shapes carry these relations (`build_surface`, `logical_surface_model_v2.py:745`):

- **bridge field** (default) — `interval_kernel × (0.75 + 0.25·cos(π·(x−center)/length))`, clamped to `[0.5, 1.0]`. A flat-top with a mild arch — connects the two spans without overpowering them.
- **scope field** (negation / uncertainty / conditional) — pure `interval_kernel` over the scope span, no arch.
- **branching field** — `interval_kernel × sin(π·(x−start)/length)` — explicit two-peak bridge that vanishes at the endpoints, so a fork looks like a fork.

#### New metric: `semantic_stability`

```
semantic_stability =
    1 / (1 + roughness
           + tension
           + negative_density
           + 0.10 · contrast_relation_pressure
           + 0.10 · uncertainty_scope_pressure)
```

A bounded `(0, 1]` score that drops when the surface is rough, when positive and negative regions overlap (tension), when negation accumulates, and when contrast/uncertainty relations pile up. It's the closest single number to "this paragraph hangs together."

#### Interpretation pipeline

`interpret_metrics` (`logical_surface_model_v2.py:1022`) takes `logical_balance_ratio`, `logical_tension_density`, `relation_to_operator_energy_ratio`, and `semantic_stability` and emits one English line per axis. The output of a v2 run is always: events table → relations table → metrics → interpretation line.

### Results — same canonical sample text

Reusing the v1 sample (77 tokens) so the v2 vs. v1 comparison is one-to-one.

Saved plots: `outputs/Figure_3.png` (total 3D surface), `outputs/Figure_4.png` (total heatmap), `outputs/Figure_5.png` (operator-only heatmap), `outputs/Figure_6.png` (relation-only heatmap).

Inferred relations on the canonical text:

```
implication:        if_scope -> then_scope
causal_support:     because_cause -> effect
contrast:           left_claim <-> but_right_claim
conclusion:         premises -> therefore_conclusion
negation_scope:     not_scope
uncertainty_scope:  maybe_scope / possibly_scope
branching:          left or right
```

Headline metrics (the new ones that didn't exist in v1):

```
relation_to_operator_energy_ratio   5.432865
scope_coverage_ratio                0.987013
semantic_stability                  low
interpretation: long-range relations dominate over isolated operators;
                low semantic stability
```

### Interpretation

The two new layers visibly do their jobs:

- The **operator-only** heatmap (`Figure_5.png`) still shows the sharp local events from v1 — `because`, `therefore`, `however`, the negation/uncertainty cluster around tokens 46–50.
- The **relation/scope** heatmap (`Figure_6.png`) shows the long-range structure: a wide `condition` bridge from `if` to `then`, a `cause` bridge from `because` to its consequent, a `contrast` bridge across `however` and `but`, and scope plateaus over the `not` and `maybe/possibly` regions.
- The **total** surface (`Figure_3.png` + `Figure_4.png`) is the superposition: locally sharp, globally connected.

The model is doing what we asked. **But the diagnostic numbers reveal a calibration problem the v1→v2 jump was supposed to expose:**

> **Relations should bend the surface, not flood it.**

A `relation_to_operator_energy_ratio` of 5.43 means the relation field carries about 5.4× the energy of the operator field. The total surface is mostly *relations*; the actual operator words are no longer the dominant geometric signal. The `scope_coverage_ratio` of 0.987 says 98.7% of the text is inside at least one relation span — for a 77-token paragraph that is almost certainly overreach.

#### Specific failure modes spotted in the inferred relations

1. **`because` inside an `if … then …` overreaches.** In:
   > "However, if the same text also contains strong causal operators, **because** it explains why one claim follows from another, then the logical flow becomes more structured."
   the human reading is: the `because` clause is a parenthetical that supports "*strong causal operators*", living *inside* the `if`-arm. The v2 rule blindly treats the right of `because` as cause and the left of `because` as effect, producing a `causal_support` whose source and target straddle the much larger `implication` span. The two relations stack instead of nest.

2. **`or` as list, not as branching.** In:
   > "… repeatedly says maybe, possibly, **or** not, …"
   the `or` is enumerating items in a list of hedge/negation markers, not opening a logical fork between two semantic alternatives. v2 still labels it `branching` and adds a `sin`-shaped bridge it does not deserve.

3. **Adjacent uncertainty scopes smear into one.** `maybe` and `possibly` sit one token apart; both create 8-token-forward scopes that overlap by ~7 tokens. Their fields add, producing a wider, deeper uncertainty trough than the text justifies.

So the v2 *prototype* is successful — the relational layer was the right structural move — but the *calibration* of the prototype is overshooting. Verdict, in one line: **successful prototype, overextended scope geometry.**

### Limitations of v2 (and what they motivate)

1. **Relation field is uncapped.** `relation_Z` can be arbitrarily larger than `operator_Z` because the two are summed without normalization. The model is supposed to allow relations to dominate when they should (a formal proof, for instance) and not dominate when they shouldn't (a paragraph that just uses local connectives). Without an energy cap, this distinction is impossible to read off the surface. → **v3 should bound `relation_Z` relative to `operator_Z`.**

2. **Scopes are not bounded by minor punctuation.** v2 stops scopes at sentence boundaries but does *not* stop them at commas, semicolons within the sentence, or at the next major operator. A `because` clause embedded in an `if … then …` is allowed to span the whole `then`-arm. → **v3 should clip scopes at the nearest comma or major operator boundary, not just the sentence end.**

3. **No notion of operator nesting.** Relations are emitted *per operator event* with no awareness of other operators that already claim overlapping spans. An `if … then …` does not know that a `because` is nested inside its condition arm. → **v3 should detect nesting and demote inner relations to "local support" rather than full bridges.**

4. **`or` is a single category.** v2 collapses `list_or` and `branching_or` into one. → **v3 should split disjunction by surrounding context: `or` flanked by hedges/negations is `list_or`; `or` flanked by clause-shaped arguments is `branching_or`.**

5. **Relations have no confidence.** Every inferred relation contributes at full amplitude. A `therefore` at the start of a sentence with no clear premises in the previous sentence contributes exactly as much as a textbook `therefore` with a long premise chain in front of it. → **v3 should attach a `confidence` to each relation and multiply amplitude by confidence.**

6. **No `list_or`-style detection for repeated uncertainty.** Three uncertainty tokens in a 4-token window should not produce three full-width overlapping scopes. → **v3 should merge adjacent same-family scopes.**

### Concrete v3 entry conditions (carrying forward from v2)

To move from v2 to v3 cleanly, the following must be true after v3:

```
relation_to_operator_energy_ratio   in [0.8, 1.5] for the canonical sample
scope_coverage_ratio                falls below ~0.70 for the canonical sample
semantic_stability                  rises (less smear ⇒ less destabilization)
```

Mathematical entry point for the cap (the simplest possible version):

```python
max_ratio = 1.25
op_e = float(np.mean(operator_Z ** 2))
rel_e = float(np.mean(relation_Z ** 2))
if rel_e > op_e * max_ratio:
    scale = math.sqrt((op_e * max_ratio) / max(rel_e, 1e-12))
    relation_Z *= scale
```

Apply after `build_surface` accumulates both layers, before `total_Z = operator_Z + relation_Z`. The cap is intentionally a soft normalization, not a hard truncation — it should never zero `relation_Z` out, only bring it back into the same order of magnitude as `operator_Z`.

Geometric entry point for thinner bridges:

```
bad   (current): interval_kernel fills everything between source and target
better (v3):     centerline field — narrow ridge along (start + end)/2,
                 falling off transverse to the X-axis bridge
```

The candidate kernel is `gaussian_along_bridge_normal × scaled_arch_along_bridge`. The arch already exists; the missing piece is a narrow transverse Gaussian so the bridge is a *ridge*, not a *plateau*.

### Open questions to revisit at v3

- Should `semantic_stability` distinguish "destabilized by genuine contradiction" from "destabilized by uncertainty smear"? Right now they look the same numerically.
- Should `argumentative_depth` count overlapping relations of *different* kinds, or only nested relations of compatible kinds (e.g., a `causal_support` nested inside an `implication` is real depth; two unrelated overlapping `contrast` and `branching` are not)?
- Is the right next step v3 (calibration of the heuristic) or a jump to v4 (replace the heuristic scope inference with a real dependency/constituency parser)? Both are reasonable. v3 is cheaper and tests whether the *shape* of the model is right; v4 tests whether the *operators* are right.

---

## Stage 3 — Bounded scopes, confidence, energy normalization (`logical_surface_model_v3.py`)

**Status:** current. Entry point: `uv run logical_surface_model_v3.py` (with `--text`, `--file`, `--no-plots`, `--no-save`, `--output-dir`, `--run-id`, `--x-resolution`, `--y-resolution`, `--operator-cross-family-width`, `--relation-cross-family-width`, `--max-relation-ratio`).

### Goal

Fix the calibration failure Stage 2 exposed — `relation_to_operator_energy_ratio ≈ 5.43`, `scope_coverage_ratio ≈ 0.987` — without abandoning the relational layer. Three things had to change:

1. **bound** the scopes so they don't spill across the whole sentence,
2. **score** each relation with a confidence and multiply amplitude by it,
3. **cap** the relation field's energy relative to the operator field.

A fourth, structural change: every run becomes a **self-contained reproducible artifact** under `outputs/v3/runs/{run_id}/` (`params.json` + `input_text.txt` + `metrics.json` + 5 PNGs).

### Model

#### Diff against v2 (only what changed)

| # | Decision | v2 | v3 | Why |
| - | -------- | -- | -- | --- |
| 1 | Tokenization | sentence-aware (`.!?;` advance sentence; `,` ignored) | clause-aware: `,` and `:` advance a `clause_index`; `.!?;` advance both; `;` advances both | scope inference needs sub-sentence boundaries — a `because`-clause embedded in a `then`-arm should not spill across the whole `then` |
| 2 | Token model | `Token(text, index, char_start, char_end, sentence_index)` | adds `clause_index`; a `ClauseBoundary(token_after, sentence_index, clause_index_before, reason)` log records every boundary that was crossed | relations now query both granularities |
| 3 | Scope endpoints | `clause_end ≡ sentence_end` | `bounded_end(default_end, max_end, event, events, max_tokens, stop_at_major_operator)` — clips at clause end, at the next major-operator start (`condition` / `cause` / `contrast`), and at a per-rule absolute token cap | v2's "everything between operator and `.`" rule was the actual source of the flooding |
| 4 | Relation object | no confidence field | adds `confidence`, `raw_amplitude`, `bridge_width`, `evidence`; `amplitude = raw_amplitude × confidence` | a fronted `because` with weak premises should not contribute as much as an explicit `if … then …` |
| 5 | `or` classification | always `branching` | `is_list_or` checks ±3-token window for uncertainty/negation markers and demotes to `list_disjunction` (lower amplitude, narrower kernel, no fork oscillation) | `"maybe, possibly, or not"` is a list, not a semantic fork |
| 6 | `because` inside `if … then …` | full causal bridge from after-`because` to sentence start | detected and demoted to `because_local_supports_if_scope` — source = after-`because` → before-`then`, target = after-`if` → before-`because` | the `because`-clause is parenthetical to the `if`-arm, not the whole sentence |
| 7 | Bridge geometry | `interval_kernel × (0.75 + 0.25·cos)` — flat-top with mild arch | `bounded_bridge_kernel = x_gate × (0.35 + 0.65·sin) × gaussian_centerline × y_gate` — a ridge along the span midpoint that fades transversely | v2's plateau was the geometric source of the flooding; v3 is a *bridge*, not a *blanket* |
| 8 | Relation energy | uncapped — `total_Z = operator_Z + relation_Z` | `normalize_relation_energy` rescales `relation_Z` so `mean(relation_Z²) ≤ mean(operator_Z²) × max_relation_to_operator_ratio` (default 1.25) | the single biggest v2 failure; without it, v2 reported 5.4× operator energy |
| 9 | Surface bundle | `operator_Z`, `relation_Z`, `total_Z` | adds `relation_Z_raw` (pre-cap), `relation_energy_scale`, so the cap is auditable | every run reports both raw and capped relation energy and the scale factor that was applied |
| 10 | Metrics | grid-independent densities, `relation_to_operator_energy_ratio`, `semantic_stability` | adds `relation_to_operator_energy_ratio_raw`, `relation_energy_scale`, `bridge_coverage_ratio`, `average_relation_confidence`, `minimum_relation_confidence`, `list_disjunction_density`; reweights `semantic_stability` to penalize *uncapped* overreach, not capped overreach | the new metrics make the v2 → v3 improvement directly measurable |
| 11 | Reporting | events + relations + metrics + 4 PNGs | adds a **raw** relation heatmap alongside the **normalized** one (5 PNGs total) | the cap must not be silent — both fields are written |
| 12 | Output layout | optional `--save-prefix` writes ad-hoc filenames | every invocation writes `outputs/v3/runs/{run_id}/` containing `params.json` (config + input SHA256 + operator-catalog SHA256 + intensifier/hedge lists), `input_text.txt`, `metrics.json`, and 5 PNGs; `run_id` is `YYYYMMDDTHHMMSSZ_<text-sha6>_<uuid4>` unless `--run-id` overrides | a metrics number with no record of the catalog, hedge set, and exact text is useless six months later |

Everything not in this table is preserved from v2 — the operator catalog *structure*, polarity convention, longest-pattern-first matching with `occupied` mask, Mexican-hat wavelet for operator events, `local_context_factor`. The catalog *values* were retuned in passing (e.g., `not` raw strength 1.00 vs. v1's 0.80, `because` 1.10) but these are calibration knobs, not load-bearing changes.

#### Bounded-scope rule (`bounded_end`, `logical_surface_model_v3.py:429`)

The single most important new helper. Every scope endpoint in v3 routes through it:

```python
end = min(default_end, max_end)               # clause clamp
if stop_at_major_operator:
    end = min(end, next_major_operator_start) # operator clamp
if max_tokens is not None:
    end = min(end, event.token_end + max_tokens)  # absolute cap
return max(event.token_end, end)              # never empty
```

`MAJOR_OPERATOR_FAMILIES = {condition, cause, contrast}` — these are the families v2 was spilling across. The per-rule `max_tokens` ceiling (5–16 depending on relation kind) is the hard absolute; the clause and operator clamps are the soft constraints that usually fire first.

#### Thin bridge geometry (`bounded_bridge_kernel`, `logical_surface_model_v3.py:1063`)

```
x_gate     = interval_kernel(X, start, end, edge_width)
arch       = 0.35 + 0.65 · sin(π · (X − start) / (end − start))
centerline = gaussian((X − midpoint) / length, σ = 0.42)
y_gate     = gaussian(Y − family_y, bridge_width)          # bridge_width ≈ 0.30–0.36
field      = x_gate · arch · centerline · y_gate
```

The difference vs. v2: a narrow `centerline` Gaussian tapers the bridge transversely to its own X-span, and the per-relation `bridge_width` along Y is ~0.30–0.36 (vs. v2's `relation_cross_family_width = 0.75`). Together these turn the v2 plateau into a *thin ridge* that connects two regions without painting the area between them. Scope-style relations (`negation_scope`, `uncertainty_scope`, `conditional_scope`, `list_disjunction`) bypass the bridge and use the simpler `scope_kernel = interval_kernel × y_gate` — they have one span, not two, so no bridge is needed.

#### Energy cap (`normalize_relation_energy`, `logical_surface_model_v3.py:1105`)

```python
op_e  = mean(operator_Z ** 2)
rel_e = mean(relation_Z_raw ** 2)
if rel_e > op_e * max_ratio:
    scale = sqrt(op_e * max_ratio / rel_e)
    relation_Z = relation_Z_raw * scale
```

Soft normalization, not truncation: the relation field is preserved *in shape*, only attenuated *in amplitude*. The scale factor is stored on `SurfaceBundle.relation_energy_scale` and emitted as `relation_energy_scale` in metrics, so any value <1.0 in a future run is the diagnostic that the cap fired. For the canonical sample the scale is `1.0` — the cap is not needed because v3's bounded scopes + thin bridges + confidence weighting *upstream* already keep `relation_to_operator_energy_ratio_raw` below 1.25 (it lands at 0.328).

#### Confidence (per-relation)

| Relation construction | Confidence | Reason |
| --------------------- | ----------:| ------ |
| `if … then …` (explicit, same sentence) | 0.96 | both endpoints present |
| `because` inside an `if … then …` | 0.82 | parenthetical role inferred from local structure |
| `because` after a same-clause claim | 0.76 | clause-local construction |
| `because` fronted (sentence-initial) | 0.58 | premise is in the *next* clause, weakly attached |
| `if` without matching `then` | 0.50 | open conditional, no target |
| `contrast` (same-clause source) | 0.86 | source span is immediately to the left |
| `contrast` (previous clause / sentence) | 0.78 / 0.72 | source resolved by lookback |
| `contrast` fallback | 0.45 | sentence-start with no clear premise |
| `conclusion` (same-clause source) | 0.78 | premises immediately to the left |
| `conclusion` (previous clause / sentence) | 0.72 / 0.70 | weaker premise binding |
| `conclusion` fallback | 0.45 | weak premise context |
| `branching` vs `list_disjunction` | 0.68 / 0.82 | a list pattern is *more* certain than a fork |
| `negation_scope` | 0.88 | local rule, high precision |
| `uncertainty_scope` | 0.78 | local rule, slightly noisier |

`average_relation_confidence` and `minimum_relation_confidence` are emitted as metrics so a sparse all-fallback run is distinguishable from one with several explicit `if … then …` structures.

### Results — same canonical sample text

The 77-token paragraph from Stage 1/2, re-analyzed. Saved as `outputs/v3/runs/example/` (params, input, metrics, 5 PNGs).

15 operator events (identical to Stage 1 — the operator-detection pipeline is unchanged). 10 inferred relations:

```
implication        sent 0  clause 0   amp +0.816  conf 0.96  if_scope -> then_scope                  explicit_if_then
contrast           sent 1  clause 2   amp −0.518  conf 0.72  left_claim <-> however_right_claim      contrast_previous_sentence
implication        sent 1  clause 3   amp +0.816  conf 0.96  if_scope -> then_scope                  explicit_if_then
causal_support     sent 1  clause 4   amp +0.508  conf 0.82  because_local_supports_if_scope         because_inside_if_then
contrast           sent 2  clause 6   amp −0.518  conf 0.72  left_claim <-> but_right_claim          contrast_previous_sentence
uncertainty_scope  sent 2  clause 6   amp −0.187  conf 0.78  maybe_bounded_scope                     local_uncertainty_scope
uncertainty_scope  sent 2  clause 7   amp −0.187  conf 0.78  possibly_bounded_scope                  local_uncertainty_scope
list_disjunction   sent 2  clause 8   amp +0.118  conf 0.82  list_left or list_right                 uncertainty_or_negation_list
negation_scope     sent 2  clause 8   amp −0.380  conf 0.88  not_bounded_scope                       local_negation_scope
conclusion         sent 3  clause 10  amp +0.648  conf 0.72  premises -> therefore_conclusion        conclusion_previous_clause
```

Headline metrics (new in v3 or directly comparable to v2):

```
relation_to_operator_energy_ratio_raw   0.328047   (v2: 5.432865)
relation_to_operator_energy_ratio       0.328047   (cap not needed; scale = 1.000)
scope_coverage_ratio                    0.740260   (v2: 0.987013)
bridge_coverage_ratio                   0.636364   (new in v3)
average_relation_confidence             0.816000   (new in v3)
minimum_relation_confidence             0.720000   (new in v3)
semantic_stability                      0.772542   (v2: "low")
logical_balance_ratio                   1.128667
logical_tension_density                 0.017958
surface_energy_density                  0.115343
```

`interpret_metrics` summary:

> constructive and destabilizing forces are relatively balanced; low local logical tension; the surface is mostly driven by local operators; no severe relation overreach was detected; moderate text coverage; high average relation confidence; moderate semantic stability.

### Interpretation

The three Stage-2 → Stage-3 entry conditions, with actual outcomes:

```
relation_to_operator_energy_ratio   target [0.8, 1.5]   actual 0.328   ← under target; cap unused
scope_coverage_ratio                target < ~0.70       actual 0.740   ← above target by 0.04
semantic_stability                  target rises         actual 0.772   ← was "low" in v2
```

The relation field is no longer flooding. The operator-only heatmap (`operator_heatmap.png`) carries most of the surface energy; the normalized relation heatmap (`relation_normalized_heatmap.png`) adds bounded ridges along the `implication`, `causal_support`, `contrast`, and `conclusion` spans without painting the whole sentence. The raw relation heatmap (`relation_raw_heatmap.png`) is identical to the normalized one for this text — diagnostic confirmation that the cap was *not* the mechanism doing the work; the bounded scopes + thin bridges + confidence weighting upstream were.

The two relations that did the heavy lifting in v2 — the two `implication`s — are now confidence 0.96, amplitude +0.816, and *spatially compact*. The `because` that v2 stacked redundantly on top of the inner `if … then …` is now correctly nested as `because_inside_if_then` with reduced amplitude (+0.508 vs. v2's ~+0.88) and a span clipped to the `if`-arm. The `or` that v2 fork-shaped is now `list_disjunction` with the lowest amplitude in the run (+0.118). These are the four concrete v2 failure modes spelled out at the end of Stage 2, all addressed.

What v3 also reveals — invisible until the surface stopped flooding — is a different class of problems: **the inferred relations are now spatially reasonable, but several are still semantically wrong.**

### Limitations of v3 (and what they motivate)

1. **`therefore` collapses target onto source.** In the canonical run the `conclusion` relation has source span and target span both equal to *"the surface develops uncertainty troughs and contradiction pressure"* — but the correct target is the post-`therefore` clause *"this geometric representation can help us compare texts by their logical shape rather than only by their words"*. Root cause (`logical_surface_model_v3.py:843`): when `therefore` is sentence-initial, `target_end = bounded_end(default_end=clause_end, …)`, and the clause containing `therefore` is just the token `therefore` itself (the immediately following comma starts a new clause). So `target_end == event.token_end`, the target span is empty, and `make_relation` falls back to copying source. → **v4 should resolve the conclusion target via `next_clause_bounds` (with `sent_end` as the fallback), not via `clause_end`.**

2. **`not` in a list is still emitted as an active `negation_scope`.** The v3 `is_list_or` check correctly demotes the `or` in *"maybe, possibly, or not"* to `list_disjunction`, but the `not` token continues down its own path and emits a separate `negation_scope` (amplitude −0.380). The model double-counts: once as part of the list, once as an active negation. → **v4 should propagate the list-context decision: a `not` / `maybe` / `possibly` that sits inside a `list_disjunction` source-or-target span should not also emit its own scope.**

3. **Mentioned-vs-used is not distinguished.** The model treats *"the text says maybe"* identically to *"maybe X"*. In the canonical sample, *"when the text repeatedly says maybe, possibly, or not"* is *talking about* the operator words, not using them — yet v3 emits two `uncertainty_scope`s and a `negation_scope` for tokens that are arguably nouns in this sentence. → **v4 should add a metalinguistic-mention detector: an operator token immediately preceded by a reporting verb (`says`, `mentions`, `writes`, `contains`, `uses`) or appearing inside quotes should be reclassified as `mentioned_operator` and not produce a scope or wave.**

4. **`if … then …` source span has no internal clause clamp.** The second implication's source is *"the same text also contains strong causal operators because it explains why one claim follows from another"*. The `because`-clause is correctly handled by rule 6 (it gets its own bounded relation), but the *outer* `if`-source still extends through the comma into the `because`-clause's territory because the source span is "after `if` → before `then`" with no `bounded_end` between them. → **v4 should clamp the `if`-source at the first major-operator boundary inside the `if`-arm (analogous to how target spans already use `bounded_end`).**

5. **The surface is a scalar field; relation topology is invisible on it.** A bridge between span A and span B looks identical to a wave centered between them. The heatmaps show *where* relation pressure sits, not *which span supports which*. → **v4 should add a relation-graph overlay: arrows or curves between source-center and target-center, drawn on top of the heatmap, colored by relation kind, line-weighted by `confidence × |amplitude|`.** Presentation change, not a model change, but it would convert the metrics-and-PNGs output into something a reader can actually *read*.

6. **`semantic_stability` may be too high on a contradictory paragraph.** The canonical text has explicit `however`, `but`, `not`, and a stacked contrast across sentences, yet scores 0.772 — solidly "moderate", almost "high". That reflects the v3 reweighting (less penalty for capped overreach), but it may have over-corrected. → **v4 should sanity-check the metric on a deliberately contradictory text (a known argument-vs-counterargument pair) and recalibrate if the metric saturates.**

### Concrete v4 entry conditions

To move from v3 to v4 cleanly, the following should hold after v4:

```
canonical conclusion target          = post-therefore clause, not a self-copy
canonical not-in-list                = no separate negation_scope emitted
mentioned-operator detection rate    ≥ 90% on a hand-labeled "says X" / "writes Y" set
relation-graph overlay               renders on every heatmap (or replaces one of the 5 PNGs)
semantic_stability                   differentiates contradictory from coherent texts
```

The first two are mechanical fixes — both are isolated boundary-arithmetic bugs in `infer_logical_relations`. The third is a small new rule (lookback for reporting verbs, plus a quote-marker pass over the tokenizer). The fourth is a new plotting layer. The fifth is calibration work, not a structural change.

### Open questions to revisit at v4

- Should `mentioned_operator` tokens be filtered out entirely (no wave, no scope), or kept on the surface with flat-zero amplitude so they still occupy token positions and don't shift the X-axis layout of other events? The latter is more consistent with the "the surface is a function of the whole text" framing.
- Should the relation-graph overlay replace one of the current 5 PNGs (e.g., merge the raw + normalized relation heatmaps into a single panel with an arrow overlay) or add a 6th? More artifacts per run is more inspectable but less navigable.
- Is `bounded_end`'s "stop at next major operator" the right granularity, or should it be "stop at next operator *of the same family*"? The current rule prevents a `because` from spilling into a `however` (right) but also prevents a `because` from spilling into a `but` (arguably wrong if the `but` is inside the `because`-clause's natural scope).
- Should `argumentative_depth` (max overlap of relation spans) be split into `homogeneous_depth` (overlapping relations of compatible kinds — e.g., `causal_support` nested inside `implication`) vs. `heterogeneous_overlap` (e.g., a `contrast` overlapping a `branching`)? Stage 2 already flagged this as open; v3 added no new evidence either way.

---

## Stage 4 — Mention detection, conclusion fix, relation-graph overlay (`logical_surface_model_v4.py`)

**Status:** current. Entry point: `uv run logical_surface_model_v4.py` (CLI shape identical to v3 — `--text`, `--file`, `--no-plots`, `--no-save`, `--output-dir`, `--run-id`, `--x-resolution`, `--y-resolution`, `--operator-cross-family-width`, `--relation-cross-family-width`, `--max-relation-ratio`).

### Goal

Address four of the six v3 limitations: (1) the `therefore` target collapse, (2) the unpropagated list-context for `not` in *"or not"*, (3) the missing mentioned-vs-used distinction, and (5) the missing relation-graph overlay. Limitations (4) the un-clamped `if`-source span and (6) the possibly-over-corrected `semantic_stability` are deliberately deferred to v5.

The conceptual addition is small but load-bearing: **a logical word being *used* and a logical word being *mentioned* are not the same event.** `Maybe the model is wrong` and `The text says maybe` produce identical token sequences for `maybe` but the second is talking about the word, not deploying it. v3 collapsed both into the same `uncertainty_scope`. v4 separates them.

### Model

#### Diff against v3 (only what changed)

| # | Decision | v3 | v4 | Why |
| - | -------- | -- | -- | --- |
| 1 | `OperatorEvent` schema | (phrase, family, span, polarity, strength, width, context_factor, amplitude, sentence_idx, clause_idx) | adds `usage` ∈ {`active`, `mentioned`, `list_connector`}, `usage_confidence`, `usage_evidence`, and `raw_amplitude` (pre-damping). `amplitude = raw_amplitude × amplitude_multiplier` where the multiplier comes from the usage classifier | every detected operator now carries *what* it is being used as, not just *what word* it is |
| 2 | Use classification | (none — every detection treated as active) | new `classify_operator_usage` (`logical_surface_model_v4.py:565`) with four passes: quote-marker check → mention-verb + (textual-subject OR uncertainty/negation phrase) → noun-of-words antecedent (`word`/`phrase`/`term`/`operator`/`marker`/`token` + no intervening punctuation) → local marker-list (≥2 hedge/negation tokens nearby with a mention verb in window) | each pass corresponds to a distinct mention pattern; falls through to `active` at confidence 0.82 |
| 3 | Amplitude damping | (none) | `MENTION_USAGE_DAMPING = 0.12` for `mentioned`, `LIST_CONNECTOR_DAMPING = 0.22` for `list_connector` | mentioned words still occupy token positions (preserving the X-axis layout of other events) but contribute ~12%/22% of the wave a true active use would |
| 4 | Mention vocabularies | (none) | three new sets: `MENTION_VERBS` (`says` / `mentions` / `writes` / `quotes` / `lists` / `names` / `calls` / `spells` …), `MENTION_NOUNS` (`word` / `phrase` / `term` / `operator` / `marker` / `token`), `TEXTUAL_SUBJECTS` (`text` / `sentence` / `paragraph` / `claim` / `argument` / `document` / `line` / `clause` / `surface` / `model` / `example`) | the rules are local and lexical — no parser, no embeddings; the three sets are the entire "knowledge" of metalinguistic mention |
| 5 | `metalinguistic_operator_mention` | (no such relation kind — mentioned operators produced active `negation_scope` / `uncertainty_scope`) | a `mentioned` event short-circuits the normal per-family branch in `infer_logical_relations` and emits a `metalinguistic_operator_mention` relation with `raw_amplitude = event.raw_amplitude × 0.12`, width 2.0, bridge_width 0.24, on the operator's original family row | the surface still records that something happened at this token position, but at ~12% the energy of an active use and with a distinct kind that downstream metrics can filter |
| 6 | Conclusion target | `target_end = bounded_end(default_end=clause_end, …)` — empty when `therefore` is sentence-initial (its clause is just `therefore`) | new helper `span_after_fronted_operator` (`logical_surface_model_v4.py:448`) — if the operator's clause is too short to hold a target, fall through to `next_clause_bounds_same_sentence`, else `sent_end`. Plus a new conclusion branch `fronted_sentence_marker` (`logical_surface_model_v4.py:1027`) that uses the *previous sentence* as source when the marker opens a new sentence | both halves of the v3 bug fixed: the source no longer falls through to the wrong clause, and the target no longer collapses onto the source |
| 7 | `is_list_or` | checked ±3-token window for uncertainty/negation phrases | also accepts neighbors whose `usage` is already `mentioned` or `list_connector`, and accepts `event.usage == "list_connector"` directly | the list-context decision now propagates from the classifier instead of being re-derived per call site |
| 8 | `list_disjunction` span | source/target = ±3 tokens from `or` (could land on filler words) | source/target = the `mentioned` events on either side of `or`, within 5 tokens — so for *"says maybe, possibly, or not"* the span becomes literally `(maybe, possibly) ↔ (not)` | the list-relation now points to the *actual list items*, not a heuristic window |
| 9 | Plotting | 5 PNGs per run (3D surface + total / operator / relation_raw / relation_normalized heatmaps) | adds `plot_relation_graph` (`logical_surface_model_v4.py:1872`) → `relation_graph.png` (6th PNG): scatter of operator centers (dimmed for `mentioned`), curved arrows between source-center and target-center for bridge-style relations (line width = `1.1 + 2.0 × min(1, |amplitude|)`, alpha = `0.55 + 0.35 × confidence`, curve direction = sign of amplitude), short horizontal bars for scope-style relations | the scalar surface shows where logical pressure sits; the graph shows which span supports which |
| 10 | `params.json` | records `intensifiers`, `hedges`, plus operator-catalog SHA256 | also records `mention_verbs`, `mention_nouns`, `textual_subjects` | the mention-classifier vocabularies are part of the model spec; a run without them recorded is non-reproducible |
| 11 | Metrics added | — | `active_operator_count`, `mentioned_operator_count`, `list_connector_operator_count`, `active_operator_density_per_100_tokens`, `mentioned_operator_density_per_100_tokens`, `metalinguistic_mention_density` | the new classifier produces categorical buckets that need to be visible in metrics |
| 12 | `interpret_metrics` | one line per axis (balance / tension / relation / normalization / coverage / confidence / stability) | adds a clause: *"some logical words are mentioned rather than actively used"* when `mentioned_operator_count > 0` | the metalinguistic regime needs to surface in the human-readable line |

Everything not in this table is preserved from v3 — operator catalog structure, longest-pattern-first matching, `bounded_end`, `bounded_bridge_kernel`, `normalize_relation_energy`, the surface bundle (`operator_Z` + `relation_Z_raw` + `relation_Z` + `total_Z`), the run-folder output layout, and the entire `if … then …`, `because`, `contrast`, `branching`, `negation_scope`, `uncertainty_scope` rule set for *active* events.

#### `classify_operator_usage` — the four passes (`logical_surface_model_v4.py:565`)

```
1. quoted_operator_word               token sits next to ", ', `, “, ” …
                                      → mentioned (conf 0.92, damp 0.12)
2. text_says_or_mentions_operator_word a MENTION_VERB in the left-6 window
                                      AND (a TEXTUAL_SUBJECT in left-6+local OR
                                           the operator is in UNCERTAINTY/NEGATION)
                                      → mentioned (conf 0.88, damp 0.12)
                                      → list_connector if BRANCHING phrase (conf 0.84, damp 0.22)
3. operator_word_named_as_term         immediately preceded by a MENTION_NOUN
                                      with no intervening punctuation
                                      → mentioned (conf 0.78, damp 0.12)
                                      → list_connector if BRANCHING (conf 0.72, damp 0.22)
4. logical_marker_list_mention         ≥2 UNCERTAINTY/NEGATION phrases in the local
                                      window AND a MENTION_VERB present
                                      → mentioned (conf 0.86, damp 0.12)
                                      → list_connector if BRANCHING (conf 0.86, damp 0.22)
0. active_logical_use                  default — no mention signal found
                                      → active (conf 0.82, damp 1.0)
```

Pass 2 is the load-bearing rule for the canonical text. *"when the text repeatedly says maybe, possibly, or not"* matches `says` (MENTION_VERB) in left-6, `text` (TEXTUAL_SUBJECT) in left-6, and `maybe` ∈ UNCERTAINTY_PHRASES — so `maybe` becomes `mentioned`, `possibly` becomes `mentioned`, `not` becomes `mentioned`, `or` becomes `list_connector`.

#### Conclusion fix — `span_after_fronted_operator` (`logical_surface_model_v4.py:448`)

```python
start = event.token_end                # after the operator
end   = clause_end                     # default: rest of the clause
if end <= start:                       # the operator is alone in its clause
    next_clause = next_clause_bounds_same_sentence(...)
    if next_clause is not None:
        start, end = next_clause       # use the next clause in this sentence
    else:
        start, end = event.token_end, sent_end  # fall back to sentence end
if max_tokens is not None:
    end = min(end, start + max_tokens)
```

Plus a new branch in the conclusion rule: when `event.token_start <= sent_start + 1` (the operator is the first token of its sentence), the source is the *previous sentence*, not the empty pre-operator span. Together these turn the v3 self-copy into a correctly-directed bridge.

#### `plot_relation_graph` (`logical_surface_model_v4.py:1872`)

Encoding:

| Visual | Meaning |
| ------ | ------- |
| scatter point at (token_center, family_y) | operator event; size 34 for `active`, 22 for `mentioned`, alpha 0.85 / 0.45 |
| rotated label above scatter | operator phrase |
| curved arrow (`arc3,rad=±0.20`) source-center → target-center | bridge-style relation (`implication`, `causal_support`, `conclusion`, `contrast`, `branching`); curve up for positive amplitude, down for negative |
| line width | `1.1 + 2.0 × min(1, |amplitude|)` |
| arrow alpha | `0.55 + 0.35 × confidence` |
| short horizontal bar at family row | scope-style relation (`negation_scope`, `uncertainty_scope`, `conditional_scope`, `list_disjunction`, `metalinguistic_operator_mention`) |
| relation-kind label | text along the arrow or below the bar |

The surface tells us *where* logical pressure exists. The graph tells us *which span supports which other span* — and it shows mentioned operators visually muted, so the eye sees what is logically load-bearing.

### Results — same canonical sample text

Saved as `outputs/v4/runs/example/` (params, input, metrics, 6 PNGs).

15 operator events. New `usage` column (only the rows that changed from "all active" in v3):

```
maybe       uncertainty   mentioned        sent 2  clause 6   raw_amp −0.130   amp −0.016   evidence text_says_or_mentions_operator_word
possibly    uncertainty   mentioned        sent 2  clause 7   raw_amp −0.130   amp −0.016   evidence text_says_or_mentions_operator_word
or          disjunction   list_connector   sent 2  clause 8   raw_amp +0.068   amp +0.015   evidence connector_between_mentioned_operator_words
not         negation      mentioned        sent 2  clause 8   raw_amp −0.800   amp −0.096   evidence text_says_or_mentions_operator_word
```

The other 11 operators remain `active` at confidence 0.82, unchanged amplitudes.

10 inferred relations (relation count unchanged from v3, but the *kinds* shift — the 1 `negation_scope` + 2 `uncertainty_scope` from v3 become 3 `metalinguistic_operator_mention`):

```
implication                         sent 0  clause 0   amp +0.816  conf 0.96  if_scope -> then_scope                  explicit_if_then
contrast                            sent 1  clause 2   amp −0.518  conf 0.72  left_claim <-> however_right_claim      contrast_previous_sentence
implication                         sent 1  clause 3   amp +0.816  conf 0.96  if_scope -> then_scope                  explicit_if_then
causal_support                      sent 1  clause 4   amp +0.508  conf 0.82  because_local_supports_if_scope         because_inside_if_then
contrast                            sent 2  clause 6   amp −0.518  conf 0.72  left_claim <-> but_right_claim          contrast_previous_sentence
metalinguistic_operator_mention     sent 2  clause 6   amp −0.014  conf 0.88  mentioned_word:maybe                    text_says_or_mentions_operator_word
metalinguistic_operator_mention     sent 2  clause 7   amp −0.014  conf 0.88  mentioned_word:possibly                 text_says_or_mentions_operator_word
list_disjunction                    sent 2  clause 8   amp +0.118  conf 0.82  mentioned_items_listed_with_or          uncertainty_or_negation_marker_list
metalinguistic_operator_mention     sent 2  clause 8   amp −0.084  conf 0.88  mentioned_word:not                      text_says_or_mentions_operator_word
conclusion                          sent 3  clause 10  amp +0.738  conf 0.82  premises -> therefore_conclusion        conclusion_previous_sentence_fronted_marker
```

Two qualitative changes worth naming:

- The **`conclusion`** relation is now non-degenerate. Source = the entire previous sentence (*"But when the text repeatedly says maybe possibly or not, the surface develops uncertainty troughs and contradiction pressure"*), target = the post-`therefore` clause (*"this geometric representation can help us compare texts by their logical shape rather than only by their words"*). v3's self-copy is gone.
- The **`list_disjunction`** source/target spans now point to the literal list items: source = `(maybe, possibly)`, target = `(not)`. v3's broader window is gone.

Metrics (v3 → v4 deltas on the bucket changes only):

```
active_operator_count                       —      → 11        (new)
mentioned_operator_count                    —      →  3        (new)
list_connector_operator_count               —      →  1        (new)
active_operator_density_per_100_tokens      —      → 15.5844   (new)
mentioned_operator_density_per_100_tokens   —      →  3.8961   (new)
uncertainty_scope_density                   2.5974 →  0.0000   (was 2 active scopes; now 2 mentions)
negation_scope_density                      1.2987 →  0.0000   (was 1 active scope; now 1 mention)
metalinguistic_mention_density              —      →  3.8961   (new, replacement total)
uncertainty_density (operator-family count) 2.5974 →  0.0000   (no more "active" uncertainty operators)

relation_to_operator_energy_ratio_raw       0.3280 →  0.4077   (rose — conclusion now contributes a real bridge)
relation_to_operator_energy_ratio           0.3280 →  0.4077   (still well under the 1.25 cap)
relation_energy_scale                       1.0000 →  1.0000   (cap not needed)
scope_coverage_ratio                        0.7403 →  0.9870   (rose — see Interpretation)
bridge_coverage_ratio                       0.6364 →  0.8571   (rose — same reason)
argumentative_depth                         4      →  5        (one more overlap from the broader conclusion target)
average_relation_confidence                 0.8160 →  0.8460
minimum_relation_confidence                 0.7200 →  0.7200
semantic_stability                          0.7725 →  0.8022
logical_balance_ratio                       1.1287 →  1.3616   (positive density rose; negative fell)
positive_wave_density                       0.1194 →  0.1258
negative_wave_density                       0.1057 →  0.0924
surface_energy_density                      0.1153 →  0.1212
logical_tension_density                     0.0180 →  0.0241
```

`interpret_metrics` summary:

> constructive/inferential forces dominate; low local logical tension; relations substantially shape the surface; no severe relation overreach was detected; broad text coverage; high average relation confidence; some logical words are mentioned rather than actively used; high semantic stability.

### Interpretation

The four targeted v3 limitations are addressed:

```
(v3 limitation 1) therefore target self-copy        →  v4 conclusion has distinct source / target spans
(v3 limitation 2) not-in-list double-counted        →  v4 not is mentioned at amp −0.096 (was −0.380 active scope)
(v3 limitation 3) mentioned-vs-used not distinguished →  3 mentioned operators classified at conf 0.88
(v3 limitation 5) no relation-graph overlay         →  relation_graph.png is the 6th PNG per run
```

`logical_balance_ratio` rose from 1.13 → 1.36 — the model now reads the paragraph as *more* constructive than v3 did. The intuition: v3 was scoring three active negation/uncertainty events at full amplitude on a paragraph whose narrator was merely *describing* uncertainty words; v4 strips that false destabilization out and the constructive structures (the two `implication`s, the `causal_support`, the now-correct `conclusion`) dominate the surface. This matches the user-level reading of the paragraph — it is an explanatory paragraph about logical structure, not an expression of uncertainty.

`semantic_stability` rose 0.77 → 0.80 for the same reason. Two contradictory pressures (uncertainty and negation densities) went to zero; the metric's denominator shrank.

But two metrics moved in the "wrong" direction relative to the v3 entry conditions, and **they should not be read as regressions**:

- **`scope_coverage_ratio` 0.740 → 0.987** — the v3 number was inflated by *contrast spans across sentences* and held down by the *broken conclusion target* (which contributed only its 9-token source instead of source + a real ~17-token target). v4 keeps the wide contrast spans and adds the correct conclusion target on top, so coverage rises. The v3 entry condition "scope_coverage_ratio < ~0.70" was wrong about what v4 *should* achieve — it was a reasonable target *given the bug*, but the metric itself doesn't separate "broad coverage from many distinct, correct relations" from "broad coverage from one ballooning relation". v5 needs to revisit the metric, not the model.
- **`relation_to_operator_energy_ratio` 0.328 → 0.408** — the correct conclusion bridge adds real energy. Still well under the 1.25 cap; the relation field still does not flood. The diagnostic value of the metric is intact; only the absolute number moved.

The visual confirmation is in `relation_graph.png`: dim small markers for `maybe`/`possibly`/`not` (mentioned), full-size markers for the 11 active operators, two upward arcs along the condition row for the implications, a small upward arc for `causal_support`, two downward arcs for the contrasts, one upward arc spanning sentence 2 → sentence 3 for the conclusion, and a short bar for the `list_disjunction` and the three `metalinguistic_operator_mention`s. The topology is now readable.

### Limitations of v4 (and what they motivate)

1. **`scope_coverage_ratio` is now ambiguous.** A run with three small, distinct relations covering 0.95 of the tokens reads identically to a run with one giant bad relation covering 0.95. The metric counts coverage, not quality. → **v5 should split it: `distinct_span_coverage` (union of *non-overlapping* relation spans), `mean_relation_span_length` (so a single huge relation can be flagged), and perhaps `coverage_per_relation_kind` (so contrast-driven coverage is distinguishable from implication-driven coverage).**

2. **`if`-source clamp is still missing.** v3 limitation 4 was not addressed in v4. The second implication's source still extends through the comma into the `because`-clause's territory: source span is `"the same text also contains strong causal operators because it explains why one claim follows from another"`. The `because`-clause is correctly handled as its own relation, but it appears *inside* the `if`-source. → **v5 should clamp the `if`-source at the first major-operator boundary inside the `if`-arm (the same `bounded_end` treatment target spans already get).**

3. **`semantic_stability` is now *higher* on a contradictory paragraph than v3.** 0.80 vs. 0.77. v3 limitation 6 was deferred and is arguably worse now: removing the false-positive active scopes for `maybe`/`possibly`/`not` removed three penalties from the metric's denominator, so a paragraph featuring `however`, `but`, and an entire sentence of described uncertainty still scores "high stability". → **v5 should sanity-check the metric on a deliberately contradictory text and recalibrate. The current weighting under-counts contrast and over-trusts the absence of active uncertainty.**

4. **Mention classification is rule-based and fragile.** Four passes, three lexicons, English-only. It correctly handles *"the text says maybe"* but misses *"the word maybe is interesting"* (no MENTION_VERB in left-6), *"consider the operators: maybe, possibly, not"* (colon-separated, MENTION_NOUN beyond direct adjacency), and *"'maybe' carries hedging weight"* (single-quote already handled, but the verb is after the operator). It also has false positives: *"Alice lists her concerns"* — `lists` is a MENTION_VERB, and if a `not`/`maybe` appears in the next 7 tokens it would falsely match pass 4. → **v5 should at minimum add a parser-light dependency check (does the operator's parent in a shallow parse target a `say`/`list`/`mention` head?), or accept the fragility and add a `--strict-mention-mode` flag for users who want to disable mention detection.**

5. **`metalinguistic_operator_mention` lives on the original operator's family row.** A mentioned `maybe` still adds a (small) negative wave to the *uncertainty* family. A paragraph talking *about* uncertainty thus still elevates the uncertainty row at 12% strength, which subtly contradicts the "this is not active uncertainty" classification. → **v5 should add a 9th family — `metalinguistic` — and route all mentioned operators to it. The original family becomes the *referent* (preserved as `event.family`), and the Y-axis row becomes `metalinguistic`. Two consequences: `FAMILY_ORDER` grows (which invalidates surface-comparison-across-versions, the v1-vintage warning); and the operator-only heatmap gains a dedicated row for "the text is being meta".**

6. **Relation-graph overlay flattens parallel relations.** The two `contrast` relations on the canonical text have identical `|amplitude| = 0.518` and `confidence = 0.72`, and the encoding (line width × alpha) gives them identical visual weight on the same family row. They overlap visually even though they belong to different sentences. → **v5 should offset parallel relations along Y (a small per-relation jitter), or draw them on slightly different curves (`rad ± 0.05`), or label each arrow with its sentence index.**

### Concrete v5 entry conditions

To move from v4 to v5 cleanly:

```
if-source clamp                       second implication source ends at the comma before "because"
distinct_span_coverage                new metric, < 0.85 on canonical text
semantic_stability                    drops on a known contradictory paragraph
                                      (e.g., "A is true. However, A is false. Therefore A is false.")
metalinguistic family row             FAMILY_ORDER includes "metalinguistic"; mentioned events emit there
relation-graph parallel relations     visually distinguishable when |amplitude| and confidence match
```

The first is a one-line `bounded_end` insertion. The second + third are calibration work plus one new metric. The fourth changes `FAMILY_ORDER` (load-bearing across the codebase). The fifth is a plotting tweak.

### Open questions to revisit at v5

- Adding a `metalinguistic` row breaks Y-axis stability — every previous surface has 8 family rows and a v5 surface would have 9. Is the right move to (a) accept the break and version the layout (`SURFACE_LAYOUT_VERSION = 2`), (b) keep the row count fixed and rename an existing under-used row, or (c) introduce a second surface (`metalinguistic_Z` alongside `operator_Z` and `relation_Z`) so the family-axis layout is preserved?
- Pass 4 of `classify_operator_usage` requires a MENTION_VERB in the local window. Should it also fire on punctuation patterns like *"consider:"* or *"e.g."* that signal a list without a verb?
- Is the `list_disjunction` encoding (source/target = literal mentioned items, single span) the right shape, or should it produce N−1 pairwise relations (one per adjacent pair of items)? The single-relation form is cleaner; the pairwise form would let the relation graph show the list as a chain of arrows.
- Should the `interpret_metrics` line mention the *direction* of the regime change (e.g., "the surface energy is dominated by the *constructive* side") rather than just "constructive forces dominate"? The current one-liner is information-dense but jargon-leaning.

---
