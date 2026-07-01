# RFC 0004 — Per-class (Mondrian) conformal parity gate

- **Status:** Draft / discussion
- **Author:** Narasimhan Rengan
- **Relates to:** the TRACER paper (Rida, 2026, arXiv:2604.14531) — Limitation
  §1 (calibration→test agreement gap); issue #38 (global threshold
  under-protects minority classes); Tier 1 of
  [RFC 0002](./0002-contribution-roadmap.md).
- **Touches:** `fit/pipeline.py` (`_calibrate_threshold`, `_cp_lower`,
  `_build_accepting_stage`, `build_global`, `apply_stage`), `config.py`
  (`FitConfig`), `policy/artifacts.py`, `runtime/router.py`, `types.py`
  (`ArtifactManifest`), and the cloud `conformal_upper` surface
  (`cloud/cli.py:415`).

This RFC replaces TRACER's single **global** agreement bound with a **per-class
conformal risk-control gate**, so the parity guarantee α holds *per predicted
class* with a finite-sample, distribution-free bound — closing the paper's #1
limitation and fixing issue #38 by the same construction.

---

## 0. TL;DR

1. Today the gate certifies **one marginal number**: a Clopper–Pearson lower
   bound on teacher agreement over the *whole accepted pool*, behind a single
   scalar acceptor threshold τ (`_calibrate_threshold`, `apply_stage`).
2. A marginal guarantee is **not** a conditional one. When the accepted pool's
   class mix shifts between calibration and production, per-class agreement can
   sit well below α even though the pooled average clears it. That is exactly the
   CLINC150 gap (95.2% calibrated → 93.0% test @ α=0.95) and issue #38 (rare
   classes hide inside the average).
3. Fix: stratify calibration by **predicted class** (a Mondrian taxonomy),
   calibrate a **per-class threshold τ_c** whose class-conditional agreement
   lower bound clears α, and **defer any class that cannot be certified**
   (τ_c = +∞). Per-class certification ⟹ marginal certification, and it can't be
   broken by test-time class reweighting.
4. Two calibration backends: **Mondrian CP** (default — a drop-in per-class
   generalization of the existing `_cp_lower`) and **Learn-then-Test (LTT)** (a
   rigorous mode that also corrects the threshold-scan multiplicity the current
   code only patches with an ad-hoc 70/30 split).
5. The runtime change is one line — `accept = scores >= τ_vec[preds]` instead of
   `scores >= τ` — plus a per-class report that doubles as an interpretability
   artifact and as the proof the gap is closed.

---

## 1. Background: how the gate works today

TRACER's L2D/RSB stages accept a surrogate prediction when an **acceptor** score
clears a threshold, and defer otherwise:

- **Acceptor** (`_fit_acceptor`): a logistic regression over four confidence
  features — top-1 prob, top-2 prob, their margin, normalized entropy
  (`_accept_features`) — predicting P(surrogate prediction == teacher). Its score
  is `s(x) = _accept_scores(acceptor, probs)`.
- **Threshold calibration** (`_calibrate_threshold`): picks **one global scalar**
  τ. It scans every unique score, and among thresholds whose **Clopper–Pearson
  lower bound** on the *pooled* accepted agreement clears the target, takes the
  lowest τ (max coverage). To fight in-sample optimism from scanning the grid it
  uses a 70/30 select/verify split with a held-out point-estimate check.
- **The bound** (`_cp_lower(k, n, δ)`): an exact Beta-quantile lower confidence
  bound on a **single** binomial rate — computed over the *entire* accepted set,
  regardless of class.
- **Application** (`apply_stage`): `accept = _accept_scores(...) >= stage["threshold"]`.
  One τ for every row. The runtime router (`runtime/router.py`) just calls
  `apply_stage`, so the scalar threshold is the only decision surface.

`build_global` is the degenerate case: accept **all** rows if the pooled lower
bound clears α (no acceptor, no threshold).

So the current guarantee is: *the pooled accepted agreement, lower-bounded at
confidence δ, is ≥ α.* One number, marginal over classes.

## 2. The defect: marginal ≠ conditional

Let ĉ(x) be the surrogate's predicted class and A the accept event `s(x) ≥ τ`.
The gate controls

```
  P(pred = teacher | A)                         (marginal agreement | accept)
```

but a user who trusts the surrogate on a *class* actually cares about

```
  P(pred = teacher | A, ĉ = c)   ≥ α   for every c   (per-class agreement | accept)
```

These differ whenever agreement varies by class *and* the accepted class mix
moves. Two consequences, both already observed:

- **Calibration→test gap (Limitation §1).** A single τ certified on the
  calibration mix is applied to a production mix with a different class balance.
  Easy, high-frequency classes carry the pooled average above α while harder
  classes drift below it — the pooled number holds up better than any individual
  class. CLINC150: 95.2% → 93.0%.
- **Minority-class under-protection (issue #38).** A rare class contributes few
  rows to the pool, so its own (possibly poor) agreement is invisible in the
  average. The global threshold happily accepts it. PR #44 adds per-class
  *thresholds* but tunes them to point estimates — no finite-sample guarantee, so
  it re-introduces exactly the in-sample optimism the global gate fought.

Root cause: **the accept decision and its guarantee are both global.** Fixing one
without the other is a half-measure.

## 3. Proposal — per-class conformal risk control

### 3.1 Condition on the *predicted* class

The taxonomy must be observable at inference time, so we stratify on the
**surrogate's predicted class** ĉ(x), not the (unknown) teacher label. This is
the Mondrian conformal construction: a separate calibration bucket per predicted
class, each certified independently.

Honest limitation up front: conditioning on ĉ gives coverage *conditional on the
predicted label*, not full conditional coverage P(·|x) — which is provably
impossible distribution-free (Vovk; Barber et al., "limits of distribution-free
conditional predictive inference"). Predicted-class conditioning is the strongest
*achievable* stratification and is exactly what fixes §1/#38.

### 3.2 Calibration backends

For predicted class c, let the class-c accepted set at threshold t have n_c(t)
accepted rows and k_c(t) teacher-agreements.

**Backend A — Mondrian CP (default).** Per class, pick the smallest τ_c (max
coverage) such that `_cp_lower(k_c, n_c, δ) ≥ α` and `n_c ≥ min_certify_n`. This
is literally the existing `_cp_lower` applied within each stratum — minimal new
surface, cheap, and monotone in α exactly as today.

**Backend B — Learn-then-Test (rigorous mode).** Treat each candidate τ_c as the
hypothesis `H: R_c(τ_c) > 1−α`, where R_c is the class-c disagreement rate.
Compute a valid p-value from the binomial tail (Hoeffding–Bentkus), and select
the accepting region by **fixed-sequence testing** along the τ grid. This is the
principled cure for the threshold-scan multiplicity that Backend A and today's
code only mitigate with the 70/30 heuristic — with LTT we can **drop the ad-hoc
held-out split** and keep a provable guarantee. Because Mondrian strata are
disjoint, each class runs LTT independently at level δ (per-class validity); a
*simultaneous* all-classes statement uses a Bonferroni split δ/C.

Recommendation: ship **Backend A as default** (drop-in, matches existing
behavior at C=1), **Backend B behind `gate_confidence_mode="ltt"`** for tight α
where multiplicity bites.

### 3.3 Defer the uncertifiable classes (the #38 fix)

If no τ_c certifies class c at (α, δ, min_certify_n) — too few calibration rows,
or agreement genuinely below α — set **τ_c = +∞**: the surrogate never handles
class c, everything defers to the teacher. This is the correct protection for
minority/hard classes: they are *deferred, not silently under-served*. The
per-class report (§4) names every deferred class and why.

Theorem (informal): if each certified class has class-conditional accepted
agreement ≥ α at confidence δ, then (i) the marginal accepted agreement ≥ α, and
(ii) the bound is invariant to the production class mix — closing the gap by
construction.

### 3.4 What changes in code

Precise, surgical change list:

| Location | Change |
|---|---|
| `_calibrate_threshold` | Return a **per-class** result: `thresholds: {class_idx: τ_c}`, plus per-class `n_cal`, `agreement_lower`, `coverage`, `status`. Add `gate="global"\|"per_class"` and `mode="cp"\|"ltt"`. |
| new `_calibrate_thresholds_per_class` | The Mondrian loop: bucket cal rows by `preds`, run the CP/LTT selection per bucket, default uncertifiable → +∞. |
| `_cp_lower` | Unchanged (reused per stratum). Add `_ltt_pvalue` for Backend B. |
| stage dict (`_build_accepting_stage`) | `threshold: float` → `thresholds: np.ndarray[n_classes]` (or scalar for old/global). Add `gate_version`. |
| `apply_stage` | `accept = scores >= τ` → `accept = scores >= τ_vec[preds]`; keep scalar path for backward compat (see §5). One line, and it is the **only** runtime change — `router.py` calls straight through. |
| `build_global` | Accept-all only if **every** predicted class clears α per-class; otherwise fall through (don't deploy an accept-all that hides a failing class). Directly ties off #38 at the global tier. |
| `_pipeline_cal_summary`, frontier | Add per-class agreement / coverage / certified-count to the summary. |

### 3.5 Naming: α vs δ (fix an existing collision)

The paper's **α** is the *agreement target* (`FitConfig.target_teacher_agreement`).
The code's **`alpha=0.1`** is the *confidence level* of the CP bound — a different
quantity (miscoverage δ). This collision is already confusing and gets worse with
per-class bounds. This RFC introduces **δ = `gate_confidence`** for the bound's
confidence and reserves **α** for the agreement target, keeping `alpha` as a
deprecated alias for one release.

## 4. Artifacts, reporting, and the cloud surface

Emit a **per-class certification table** into `manifest.json` / `frontier.json` /
the qualitative report:

| class | n_cal | τ_c | agreement (lower @δ) | coverage_c | status |
|---|---|---|---|---|---|
| `card_arrival` | 142 | 0.61 | 0.971 | 0.88 | certified |
| `exchange_rate` | 9 | +∞ | — | 0.00 | deferred (n<min) |

This table is three things at once: (i) the **proof** the gap is closed (per-class
test agreement ≥ α for all certified classes), (ii) a new **interpretability
artifact** in the spirit of the paper's slice summaries, and (iii) the natural
per-class source for the hosted gateway's `conformal_upper` field
(`cloud/cli.py:415`) — aligning OSS fit-time bounds with the cloud's per-query
number. Cloud wiring is noted but out of scope here.

## 5. Backward compatibility & config

- **Artifacts.** Old `pipeline.joblib` stages carry a scalar `threshold`;
  `apply_stage` detects scalar vs. vector via `gate_version`/type and keeps the
  scalar path. No forced re-fit.
- **Config.** Add to `FitConfig`: `gate: str = "per_class"` (with `"global"` to
  restore old behavior), `gate_confidence: float = 0.1` (δ, alias `alpha`),
  `gate_confidence_mode: str = "cp"` (`"ltt"` for rigorous), `min_certify_n: int
  = 10` (mirrors `cold_start.md`'s ≥10/class rule).
- **Default flip.** Propose defaulting to `per_class`, because the whole point is
  that the honest guarantee should be the default; `global` remains one flag away.
  (Open question 9.1 — could stage the flip over a release.)

## 6. Experiments / proof

Run on the paper's three datasets via the benchmark harness (RFC 0002 Tier 2):

1. **CLINC150 — the headline.** Reproduce the 95.2%→93.0% marginal gap under the
   global gate; show the per-class gate holds **every certified class** ≥ α on
   held-out test, driving the marginal gap to within statistical slack. This is
   the number that closes Limitation §1.
2. **Banking77 — #38.** Publish the per-class certified-coverage table; show
   minority classes are deferred rather than accepted below α, and report the
   coverage cost of the stricter gate honestly.
3. **MNLI — negative control.** The gate must still refuse (≈0% coverage). Per-class
   certification must not *over*-certify where frozen embeddings can't separate.
4. **Metric:** `max_c [ α − test_agreement_c(accepted) ]` (worst per-class
   violation). Global gate: positive (the gap). Per-class gate: ≤ statistical
   slack at δ. Report across seeds.

## 7. Risks & tradeoffs

- **Coverage drops.** A per-class (conditional) bound is stricter than a marginal
  one, so total coverage at fixed α falls — the honest price of the guarantee.
  Mitigations: LTT is less conservative than union-bounded CP; only genuinely
  uncertifiable classes are deferred.
- **Data hunger.** C classes × few calibration rows each ⇒ many deferred classes
  on small data. This interacts with cold start; the gate must degrade gracefully
  (defer, never crash), consistent with the current philosophy. `min_certify_n`
  makes the cutoff explicit.
- **Multiplicity.** Scanning τ per class over a grid inflates the effective error
  rate; naive per-class point thresholds (cf. #44) are *not* valid. Backend B
  (LTT / fixed-sequence, Bonferroni across classes) is the rigorous answer;
  Backend A leans on the disjoint strata + `min_certify_n` + held-out check.
- **Not full conditional coverage.** Predicted-class conditioning is the
  achievable target; we state this rather than over-claim.

## 8. Non-goals

- Per-class *acceptors* (one acceptor, per-class thresholds here); a per-class or
  richer acceptor is a possible follow-up.
- The hosted gateway's per-query conformal serving — only the OSS fit-time
  calibration and its artifacts are in scope.
- Changing the acceptor feature set or the surrogate zoo.

## 9. Open questions

1. **Default gate:** ship `per_class` as the default immediately, or stage the
   flip (`global` default for one release with a loud deprecation)?
2. **δ spend across classes:** per-class validity at δ (each class individually),
   or simultaneous validity via Bonferroni δ/C by default? The latter is stricter
   but gives a single global statement.
3. **Ground-truth mode:** when `ground_truth` is present, also report per-class
   *accuracy* bounds alongside teacher-agreement bounds (supports the paper's
   ground-truth-vs-teacher analysis, Limitation §3)?
4. **RSB interaction:** per-class thresholds are per stage; confirm the residual
   split (`build_rsb`) still leaves enough per-class calibration mass at stage 2,
   or fall back to a global stage-2 gate when it doesn't.
