# RFC 0001 — Tiered Surrogate Cascade: extending TRACER with fine-tuned encoders and small local LLMs

- **Status:** Draft / discussion
- **Author:** Narasimhan Rengan
- **Relates to:** the TRACER paper (Rida, 2026, arXiv:2604.14531) — Limitations
  §1 (calibration→test gap), §3 (mimics teacher), §6 (frozen embeddings); and
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

| Property            | Classical surrogate | Small decoder LLM (≈1B) |
| ------------------- | ------------------- | ----------------------- |
| Inference latency   | microseconds, CPU   | ~100ms–1s; GPU-preferred |
| Marginal cost       | ~free               | real (GPU amortization) |
| Refit cost (flywheel)| seconds            | minutes–hours (LoRA)    |
| Deploy artifact     | one `joblib` file   | model server + weights  |

So "small LLM **instead of** ML" weakens the headline metric. We must not frame
it as a replacement.

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
5. **Multi-task** — one tuned small LLM serving many endpoints.

### 3.2 Sketch: slotting a tier into the zoo

`search_best_surrogate()` / `_candidates()` already return scored candidates.
A new tier is a candidate that:

- exposes `predict_proba`-equivalent **calibrated** scores so the existing
  acceptor features still apply (decoder LLMs: derive from token logprobs and/or
  self-consistency; verbalized confidence is *not* enough);
- for decoder tiers, uses **constrained decoding** (outlines /
  lm-format-enforcer / logit masking to the label set) so it cannot emit an
  off-label class;
- carries a **cost tag** (est. ms + $/1k inferences incl. GPU amortization) so
  selection can prefer the cheapest tier that clears α, not just the most
  accurate.

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

Add a RouteLLM-style learned-router baseline so the comparison answers the
paper's "limited baselines" limitation at the same time.

## 6. Risks / non-goals

- **Cost moat.** Heavier tiers must be **opt-in** (`pip install
  tracer-llm[local]`), never the default, or OSS adoption suffers.
- **Calibration.** Decoder confidence is unreliable → constrained decoding +
  the per-class conformal gate are required, not optional.
- **Still mimics the teacher** (paper §3). Distillation can't beat the teacher
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
  chosen tier)?
