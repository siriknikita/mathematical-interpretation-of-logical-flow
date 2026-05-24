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
