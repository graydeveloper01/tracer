# RFC 0001 — Tiered Surrogate Cascade: extending TRACER with fine-tuned encoders and small local LLMs

- **Status:** Draft / discussion
- **Author:** Narasimhan Rengan
- **Relates to:** the TRACER paper (Rida, 2026, arXiv:2604.14531) — Limitations
  §1 (calibration→test gap), §2 (mimics teacher), §3 (frozen embeddings); and
  issues #38, #45.

This document captures a design discussion about how to extend TRACER beyond a
classical-ML surrogate on frozen embeddings, without giving up its defining
property: near-zero marginal inference cost.

---

## 0. TL;DR

1. The naive idea — "replace the ML surrogate with a small local LLM" — is the
   *wrong* framing, because it quietly trades away TRACER's headline metric
   (near-zero marginal cost) for a smaller-but-real cost.
2. The right framing is a **cost-tiered surrogate cascade**: keep the cheap
   classifier as Tier 0, add a fine-tuned encoder and (optionally) a small
   local decoder LLM as higher tiers, and let the **parity gate** decide which
   tier — if any — is trustworthy enough to handle each input before deferring
   to the teacher. This is *additive*, preserves the cost story, and turns the
   paper's MNLI negative result into a test of the new tiers.
3. A companion change — a **per-class (Mondrian) conformal parity gate** —
   composes with the cascade and fixes the paper's #1 limitation (the
   calibration→test agreement gap) and issue #38.

---

## 1. Background: what TRACER does today

TRACER trains a classical-ML surrogate (logistic regression / MLP / trees /
ensembles — `fit/surrogate.py::_candidates`) on **frozen** BGE-large-en-v1.5
embeddings of an LLM's own production traces. A pool of candidates competes at
each refit, selected by macro-F1 (`search_best_surrogate`). An **acceptor**
(logistic regression over four confidence signals: top-1 prob, top-2 prob,
their margin, normalized entropy) predicts per-input reliability. A **parity
gate** activates the surrogate only when its agreement with the teacher on a
held-out split exceeds a user threshold α; otherwise it defers to the teacher.
Deferred inputs self-label via the teacher and feed the next refit (the
flywheel).

Reported results: Banking77 100% coverage @ α=0.80 (83.2% @ α=0.95); CLINC150
full replacement; **MNLI 0% coverage — the gate correctly refuses to deploy
because frozen embeddings can't separate entailment relations.**

## 2. The core tension

TRACER's value proposition is one number: **near-zero marginal inference
cost**. The classical surrogate is a `joblib` file doing a matrix multiply in
microseconds on CPU; refitting it is seconds.

A small *decoder* LLM as the surrogate changes that:

| Property            | Classical surrogate | Fine-tuned encoder (≈100–400M) | Small decoder LLM (≈1B) |
| ------------------- | ------------------- | ------------------------------ | ----------------------- |
| Inference latency   | microseconds, CPU   | ~50–500ms single-stream, CPU   | ~100ms–1s; GPU-preferred |
| Marginal cost       | ~free               | small but nonzero (CPU)        | real (GPU amortization) |
| Refit cost (flywheel)| seconds            | minutes (head / LoRA tune)     | minutes–hours (LoRA)    |
| Deploy artifact     | one `joblib` file   | weights + tokenizer, self-contained | model server + weights |

So "small LLM **instead of** ML" weakens the headline metric. We must not frame
it as a replacement. Honesty note: **Tier 1 also breaks the microseconds
story**, just less — "least threat to the cost moat" (§7) is relative, and the
benchmark's cost model must include Tier 1's own row rather than hide it.

## 3. Proposal A — Cost-tiered surrogate cascade

The existing zoo already *selects the best candidate and lets the gate decide
whether to deploy it*. The extension is to **add tiers to the zoo and make
selection cost-aware**:

- **Tier 0** — LR/MLP on frozen BGE embeddings *(today)* — microseconds, CPU.
- **Tier 1** — **fine-tuned encoder** (DeBERTa / BGE + classification head,
  ~100–400M). Unfreezes the representation; still CPU-deployable.
- **Tier 2** — **cross-encoder** for pairwise / compositional tasks (the direct
  MNLI fix).
- **Tier 3** — **LoRA-tuned small decoder** (Qwen 0.5–1.5B, Llama 3.2 1B) for
  open / structured / generative outputs.

The parity gate then generalizes from binary *(surrogate vs. teacher)* to a
**cascade**: easy → Tier 0, medium → a small-LLM tier, hard → teacher. Crucially
**you only pay for a heavier tier on inputs the cheaper tier defers**, so the
cost story survives — it gains a middle rung instead of losing its floor.

### 3.1 Where the extra tiers are genuinely "more versatile"

Pin the vague word "versatile" to capabilities the classifier *structurally
cannot* have:

1. **Compositional boundaries** — NLI, negation, multi-hop, long context.
   Frozen embeddings can't linearly separate these (why MNLI got 0% coverage).
2. **Open / evolving label space** — a classifier is frozen to labels seen in
   traces; an LLM can handle a described-but-unseen label.
3. **Structured outputs** — JSON, multi-label, rationale strings.
4. **Cold start** — a pretrained small LLM has priors, so day-1-with-50-traces
   it may beat a from-scratch classifier. Attacks the flywheel's slowest phase.
   Capped by the gate, though: at α=0.95, δ=0.1 the Clopper–Pearson bound needs
   ~45 zero-disagreement calibration rows in the accepted pool before *any*
   tier can be certified (see RFC 0004 §7) — the gate, not the model, sets the
   cold-start floor. The zero-shot tier raises quality so certification arrives
   as early as the math allows, but not earlier.
5. **Multi-task** — one tuned small LLM serving many endpoints.

### 3.2 The cascade already exists in the code — build on it

Two extension points exist today, and they are different things:

- **Within a stage**, `search_best_surrogate()` / `_candidates()` train a zoo
  of candidates that *compete*; one winner deploys.
- **Across stages**, `build_rsb()` already implements a residual cascade —
  stage 2 trains on the rows stage 1's gate rejects — and `route_pipeline()`
  generically walks an ordered stage list, first-accept-wins. The runtime
  router calls straight through it.

Tiers must be **stages, not candidates**: cascade semantics (easy → Tier 0,
medium → Tier 1, hard → teacher) require sequential composition, whereas the
zoo picks a single winner. The clean design uses both: generalize `build_rsb`
→ `build_cascade(tiers=[...])`, where each tier is a stage trained on the
previous tier's residual with its own acceptor and parity gate, and keep the
zoo competing *within* each tier. (This also answers open question 8.3: the
codebase has already chosen per-input routing, via the RSB residual
structure.)

The per-candidate contract inside a tier stays as before. Each candidate:

- exposes `predict_proba`-equivalent **calibrated** scores so the existing
  acceptor features still apply (decoder LLMs: derive from token logprobs and/or
  self-consistency; verbalized confidence is *not* enough). Note the acceptor
  (`_fit_acceptor`) is itself a *learned recalibration layer* trained on actual
  correctness — it can absorb much of a decoder's logprob miscalibration,
  provided the tier yields a full label-probability vector (one logit-masked
  forward pass, not one scoring call per label) for the entropy/margin
  features;
- for decoder tiers, uses **constrained decoding** (outlines /
  lm-format-enforcer / logit masking to the label set) so it cannot emit an
  off-label class;
- carries a **cost tag** (est. ms + $/1k inferences incl. GPU amortization) so
  selection can prefer the cheapest tier that clears α, not just the most
  accurate.

### 3.3 Plumbing changes the cascade actually needs

The current pipeline is embedding-in end-to-end; Tier 1+ breaks that
assumption. Concretely:

- **Stage API.** `apply_stage(stage, X)` and `route_pipeline()` receive only
  embedding arrays, and `Router.predict` enforces `manifest.embedding_dim`.
  Tier 1/2/3 consume raw text, so the stage interface must carry
  `(text, embedding)` pairs; text is already available at the `Router` surface
  when an embedder is attached, but never reaches the stages.
- **Serving surfaces.** `tracer serve` and the JS integration POST
  *embeddings*; both need a text-in mode before any text-consuming tier can
  run behind them.
- **Trace schema.** Tier 2 (cross-encoder) needs premise/hypothesis as
  separate fields; traces are `{"input": str, "teacher": str}` today.
  Concatenation is fine for frozen embeddings but defeats the purpose of a
  cross-encoder — the schema needs a structured-input variant.
- **Cost-aware selection happens at two levels.** Candidate-within-stage
  selection is pure macro-F1 (`search_best_surrogate`), and the frontier's
  method selection keys on (coverage, TA, −n_stages) — neither has a cost
  term. The cost tag from §3.2 must reach both.
- **Fit-time cost, not just inference cost.** The sweep trains and evaluates
  every candidate on every refit, and `tracer.update()` refits on every
  flywheel turn — a LoRA tier in the default sweep turns a seconds-long refit
  into minutes–hours. Heavy tiers should refit **lazily**: only when Tier 0's
  certified coverage plateaus or the residual mass crosses a floor.

## 4. Proposal B — Per-class (Mondrian) conformal parity gate

Today the gate is a single **global** Clopper–Pearson lower bound
(`fit/pipeline.py::_cp_lower`) on one global threshold. The paper's #1
limitation is the calibration→test gap (CLINC150: 95.2% calibrated → 93.0% test
agreement @ α=0.95), and issue #38 is "global threshold under-protects minority
classes." Both are the same root cause.

Replace the single global guarantee with **per-class conformal risk control**
(Mondrian CP / Learn-then-Test): certify agreement ≥ α *per class* with a
finite-sample guarantee. This:

- closes the calibration→test gap with a distribution-free bound;
- protects minority classes (issue #38) by construction;
- becomes **more** necessary once decoder tiers exist, because LLM confidence is
  badly calibrated — so A and B compose.

## 5. Experiments (reuses a shared benchmark harness)

Run on the paper's exact three datasets, add Tier 1/2/3 to the zoo, and produce
a **3-way cost–coverage–quality frontier**: all-teacher vs. classical-TRACER vs.
tiered-TRACER, with GPU amortization folded into the cost model. Two questions
decide everything:

1. **Coverage:** does a small-LLM / cross-encoder tier **pass the parity gate on
   MNLI**, where frozen embeddings get 0% coverage? (If yes, the paper's
   negative result becomes positive.)
2. **Economics:** is the **cost-per-correct-deferral** of each heavier tier
   below the teacher's, after honest GPU/latency accounting?

A "no" on (2) for some task is itself a finding: it maps exactly which task
regimes justify the heavier tier.

Add a RouteLLM-style learned-router baseline. The paper's evaluation has a
single baseline (a confidence-threshold LR with full hindsight), so this
strengthens the comparison — note this addresses a *weakness*, not a stated
limitation: "limited baselines" does not appear in the paper's limitations
(its §4 is limited *task* coverage).

## 6. Risks / non-goals

- **Cost moat.** Heavier tiers must be **opt-in** (`pip install
  tracer-llm[local]`), never the default, or OSS adoption suffers.
- **Calibration.** Decoder confidence is unreliable → constrained decoding +
  the per-class conformal gate are required, not optional.
- **Still mimics the teacher** (paper §2). Distillation can't beat the teacher
  without mixing in ground truth; sell *coverage on previously-undeployable
  tasks* + cost, not accuracy gains.
- **Ops burden.** A model server is heavier than a `joblib` file; keep Tier 0
  the always-available default.

## 7. Recommended sequencing

1. **Start with the fine-tuned encoder (Tier 1/2), not the decoder LLM.** Best
   cost/quality point for closed-label classification, most direct MNLI fix,
   stays CPU-deployable — least threat to the cost moat.
2. Land the **per-class conformal gate** (Proposal B) — small, high-signal,
   independently valuable.
3. Only then add the **decoder tier (Tier 3)**, gated on the cost-per-correct-
   deferral metric from the benchmark.

## 8. Open questions

- Does the target use case actually need *generation*, or is it really "the
  frozen embedding can't separate my classes"? (The answer decides whether
  Tier 1/2 suffices or Tier 3 is needed.)
- What's the minimum trace volume at which each tier first clears α? (Informs
  the cold-start story.)
- Should tier selection be per-policy or per-input (a true cascade vs. a single
  chosen tier)? (§3.2 argues the code has already answered: per-input, via the
  RSB residual structure.)
