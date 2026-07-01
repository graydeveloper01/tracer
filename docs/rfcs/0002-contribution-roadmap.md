# RFC 0002 — Contribution Roadmap: high-impact areas to own in TRACER

- **Status:** Planning
- **Author:** Narasimhan Rengan
- **Relates to:** the TRACER paper (Rida, 2026, arXiv:2604.14531) — Limitations
  §1–§6; issues #7, #19, #21, #27, #28, #38, #45, #59, #62, #64.

This document orders the contributions I want to make to TRACER **from highest
to lowest impact**. "Impact" here means: how directly the work (a) strengthens
the paper's core scientific claim (the parity guarantee), (b) expands the set of
tasks TRACER can serve, or (c) makes the runtime/product production-grade —
weighted above cosmetic fixes.

## Positioning principle

The repo is already swarmed by drive-by contributors: nearly every open bug and
"good first issue" has a competing one-line PR (#58→#60, #59→#61/#63, #62→#66,
#64→#65, #45→#55, #38→#44, #30→#41/#67, #27/#28→#35, #21→#42). Matching that
pile is low-signal. The plan below is deliberately the **opposite**: own a whole
area end-to-end (design → implementation → tests → docs → a benchmark that
*proves* it), on work where no competing PR exists. Every claim ships with a
number.

---

## Tier 1 — Core moat: strengthen the parity guarantee

### 1. Per-class (Mondrian) conformal parity gate  ★ highest impact
- **Why #1:** fixes the paper's own **#1 limitation** — the calibration→test
  agreement gap (CLINC150: 95.2% calibrated → 93.0% test @ α=0.95) — *and*
  issue #38 (global threshold under-protects minority classes). Same root cause.
- **Gap in code today:** the gate is a single **global** Clopper–Pearson lower
  bound (`fit/pipeline.py::_cp_lower`, `_calibrate_threshold`) on one global
  threshold. `conformal-prediction` is already a `pyproject.toml` keyword —
  promised, not delivered. No competing PR does *risk control* (only #44 does
  per-class *thresholds* without a finite-sample guarantee).
- **Scope:** per-class conformal risk control (Mondrian CP / Learn-then-Test)
  certifying agreement ≥ α *per class*, distribution-free. Touches
  `_calibrate_threshold`, `build_l2d`, `build_global`, the manifest, and the
  cloud `conformal_upper` surface (`cloud/cli.py:415`).
- **Proof:** close the CLINC150 gap; report per-class certified coverage on
  Banking77/CLINC150.

## Tier 2 — Make every other contribution legible

### 2. Shared benchmark / eval harness (+ RouteLLM baseline)
- **Why #2:** the paper admits "limited baselines" (only confidence
  thresholding). A reproducible suite over the exact three datasets
  (Banking77 / CLINC150 / MNLI) + a **RouteLLM**-style learned-router baseline +
  the cost model becomes the company's evaluation backbone — every future change
  gets measured against it.
- **Also closes:** #19 (benchmark conflates embedding vs classifier latency) and
  subsumes #21/#42 (custom-dataset notebook) at a higher altitude.
- **Deliverable:** a `benchmarks/` harness emitting the paper's
  coverage–quality–cost frontier tables + an honest latency breakdown.

## Tier 3 — Expand the addressable task set

### 3. Pluggable embedding/encoder abstraction
- **Why #3:** the paper's frozen-embedding limitation (§6) and issue #45 (OpenAI
  embeddings) share a root: `embeddings/embedder.py` is not a clean interface.
  A pluggable encoder API — OpenAI embeddings, fine-tunable HF encoders, and
  cross-encoders that can separate compositional tasks (e.g. the NLI case where
  frozen embeddings got 0% coverage) — expands the set of tasks that can clear
  the parity gate. #55 only bolts on one factory method; the interface itself is
  unowned.
- **Proof:** a task that fails the gate on frozen BGE clears it under a
  fine-tuned/cross-encoder representation.

## Tier 4 — Production-grade runtime

### 4. Serving runtime rewrite
- **Why:** `runtime/serve.py` is a 149-line single-threaded `HTTPServer`
  (issues #62 serializes requests, #64 crashes on disconnect). Competing PRs
  just swap in `ThreadingHTTPServer` — a band-aid. Own the whole story: async/
  ASGI serving, **dynamic micro-batching** of inference, **batch fallback to
  teacher** (#27/#28 — today `predict_batch` returns `None` for deferrals),
  graceful shutdown, and metrics. Fold in the OOD per-query refit fix (#59:
  `runtime/router.py::_ood_flags` refits NearestNeighbors every call).
- **Proof:** throughput + p99 latency before/after under concurrent load.

## Tier 5 — Product flywheel (the business)

### 5. Cloud watch integrations + auto-optimize triggers
- **Why:** `tracer.watch` + `tracer cloud` are v0.3.0 (brand new) — this is where
  revenue lives. High-value, unowned work: more `watch` sinks/adapters
  (**LiteLLM, LangChain, DSPy, Instructor, Vercel AI SDK**), drift detection, and
  auto-retrain triggers that close the flywheel loop.

## Tier 6 — Adoption & credibility

### 6. Reproducibility, cost calculator, doc/test gaps
- End-to-end reproduce-the-paper notebook (raises #21/#42 to "trusted
  benchmark").
- Interactive cost calculator (turns the paper's $ numbers into a live tool).
- Fill test + doc gaps: serve / embedder factories / troubleshooting (#7).

---

## Impact-ordered summary

| # | Contribution | Primary payoff | Paper/issue anchor |
|---|--------------|----------------|--------------------|
| 1 | Per-class conformal gate | Strengthens the core guarantee | Limitation §1, #38 |
| 2 | Benchmark + RouteLLM baseline | Makes all work measurable | Limitation §4, #19 |
| 3 | Pluggable encoder API | Serves previously-undeployable tasks | Limitation §6, #45 |
| 4 | Serving runtime rewrite | Production-grade | #27, #28, #59, #62, #64 |
| 5 | Cloud watch integrations | Drives the product/revenue | v0.3.x cloud |
| 6 | Reproducibility & docs | Adoption & trust | #7, #19, #21 |

**Recommended first two to land:** #1 (per-class conformal gate) and #2
(benchmark harness) — together they demonstrate research depth *and* engineering
ownership, and #2 is what makes #1's result credible.
