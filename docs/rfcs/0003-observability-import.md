# RFC 0003 — Warm start: importing traces from observability platforms

- **Status:** Draft / discussion
- **Author:** Narasimhan Rengan
- **Relates to:** the TRACER paper (Rida, 2026, arXiv:2604.14531) — cold start as
  the flywheel's slowest phase; [`docs/cold_start.md`](../cold_start.md); the
  cloud flywheel tier of [RFC 0002](./0002-contribution-roadmap.md); `watch`
  (shipped 0.3.0).

This document proposes letting TRACER **backfill its training set from an
existing LLM observability platform** (Langfuse first; LangSmith, Helicone,
Arize Phoenix, Braintrust, Weave next) so a team gets a deployable routing
policy on day one instead of collecting traces from scratch.

---

## 0. TL;DR

1. Cold start is TRACER's slowest phase, and today's only answer
   (`docs/cold_start.md`) is "hand-seed 100–300 inputs and label them
   yourself." But any team already running an observability platform has been
   logging the teacher's `(input, output)` pairs for months — that history **is**
   a labeled trace set.
2. Add `tracer import <platform>`: a batch backfill that pulls historical
   traces, maps them to TRACER's `{input, teacher}` record, and hands off to
   `tracer.fit()`. Warm start replaces cold start.
3. The plumbing is ~80% built: `watch.GenAISpan.to_trace_record()` already maps
   an OpenTelemetry GenAI span to a `TraceRecord`, and TRACER already speaks the
   `gen_ai.*` schema most platforms export. The **one new hard problem** is
   turning generative observability logs into clean *classification* traces —
   i.e. **label extraction + teacher-identity filtering**.
4. Build **one importer interface + thin per-platform adapters**, with OTel
   GenAI as the pivot format. Langfuse is the first adapter.

---

## 1. Background: what TRACER ingests today

TRACER trains on a minimal record — `TraceRecord(input_text, teacher_label)`
(`types.py:10`), with optional `trace_id`, `ground_truth`, `metadata`. The JSONL
loader (`traces/loader.py`) already accepts key aliases (`query/text/prompt/
question` → input; `output/answer/label/intent/teacher_output` → teacher), so
middleware dumps load without reshaping.

Two facts make an importer cheap:

- **`watch.GenAISpan.to_trace_record()` (`watch.py:162`) is already the exact
  mapping we need:** `input.messages → input_text`, `output.messages →
  teacher_label`, everything else → `metadata`.
- **TRACER already speaks OpenTelemetry GenAI (`gen_ai.*`).** `watch` emits that
  shape and fans it out via `OTLPSink`. Langfuse, Phoenix, Weave, and Braintrust
  have converged on the same conventions for ingest/export. TRACER currently only
  *emits* the shape; this RFC teaches it to *ingest* it.

## 2. Motivation

`docs/cold_start.md` asks a prospect to *manufacture* the asset they usually
already own:

> Pick 100–300 representative inputs … Run each through your LLM … record the
> output.

For a team already on Langfuse, that asset is sitting in a database with months
of depth. Importing it changes the first-run experience from *"collect 200+
traces over weeks"* to *"`tracer import langfuse` → fit a deployable policy in
the first ten minutes."* That is the difference between a trial that ramps and
one that lands, and it directly attacks the paper's slowest-phase limitation.

## 3. Proposal — two shapes that compose

### 3.1 Batch importer (the cold-start piece — this RFC)

A one-time backfill:

```
platform history ─▶ adapter ─▶ GenAISpan ─▶ to_trace_record() ─▶ traces.jsonl ─▶ fit()
```

CLI sketch:

```bash
tracer import langfuse \
  --name intent-classifier \        # which endpoint/prompt/generation
  --input '$.input.query' \         # input extractor (JSONPath)
  --label '$.output.intent' \       # label extractor (JSONPath)
  --model claude-sonnet-4.6 \       # teacher-identity filter
  --since 90d \
  -o traces.jsonl
# then:
tracer fit traces.jsonl .tracer
```

### 3.2 Streaming (steady state — already solved)

`watch` (0.3.0) already handles the ongoing flywheel. The pitch is the pairing:
**backfill from the platform for a warm day-1 policy, then `watch` for continual
refits** — one schema end to end.

## 4. Architecture — one interface, thin adapters

Do **not** build N importers. Normalize every source to `GenAISpan` and reuse
the existing record mapping:

```
LangfuseAdapter ─┐
LangSmithAdapter ─┼─▶ normalize to GenAISpan ─▶ to_trace_record() ─▶ TraceDataset
PhoenixAdapter  ─┘        (watch.py, exists)         (exists)
```

```python
class ObservabilityAdapter(Protocol):
    def fetch(self, *, filters: ImportFilters) -> Iterator[GenAISpan]: ...

@dataclass
class ImportFilters:
    name: Optional[str] = None          # span/generation/prompt name
    model: Optional[str] = None         # teacher-identity filter
    prompt_version: Optional[str] = None
    tags: Sequence[str] = ()
    since: Optional[str] = None          # "90d", ISO date
    limit: Optional[int] = None
```

- Platforms that natively export OTel gen_ai collapse to a single
  `OTLPImporter` (read spans, done).
- The rest need a ~50-line adapter mapping their trace JSON onto `GenAISpan`.
- **Langfuse first:** largest OSS mindshare, clean `/api/public` query API,
  self-hostable (so it is trivial to spin up for integration tests).

New optional dependency group so the core stays lean:
`pip install tracer-llm[import]` (or per-platform extras).

## 5. The hard part — observability logs are not classification traces

Sections 1–4 are plumbing. The real design work is here, and it is where the RFC
wants review.

### 5.1 Label extraction (the crux)

A platform stores the generation `output` as free-form text or JSON, not a class
label. TRACER needs a discrete label. Provide a layered extractor:

1. **JSONPath / regex** for structured or semi-structured outputs
   (`$.output.intent`, `/^Label:\s*(\w+)/`).
2. **Callable** (`--label-fn module:func`) for anything bespoke.
3. **Label-space inference:** derive the class set from the extracted values,
   with a **min-frequency cutoff** to drop noise/typos, and a report of the
   inferred taxonomy for the user to confirm before fitting.

If extraction yields too many singleton labels, that is a signal the traces are
not a classification task — fail loudly with guidance, don't silently fit.

### 5.2 Teacher identity

TRACER's parity gate certifies agreement with **one** teacher `T(x)`. Historical
logs mix models, prompt versions, temperatures. Backfilling naively pollutes the
parity target with a moving teacher. Therefore:

- `--model` / `--prompt-version` filters are **first-class**, not afterthoughts.
- The importer **warns** (and prints a breakdown) when the selected set spans
  multiple models or prompt versions, and requires an explicit
  `--allow-mixed-teacher` to proceed.

### 5.3 Task selection

Most spans in a project are not the classification call (RAG retrieval,
summarization, agent steps). `--name` / `--tags` / prompt-template filters scope
the import to the right generation.

### 5.4 Data hygiene

- **Dedup** repeated identical `(input, label)` rows (common in prod logs).
- **PII / masking:** respect the source platform's redaction; never widen data
  exposure. Import is opt-in per project.
- **Volume/sampling:** platforms may sample; surface per-class counts against the
  `cold_start.md` rules of thumb (≥10/class, ≥5 classes) and warn when short.

## 6. Non-goals / risks

- **Not a replacement for `watch`.** Import is the warm start; `watch` is the
  steady-state flywheel. This RFC is only the backfill.
- **Mimics the teacher (paper §3).** Imported labels are the teacher's historical
  outputs, errors included — the surrogate learns to reproduce them, not to beat
  them. `cold_start.md` should say so plainly.
- **Adapter drift.** Platform APIs change; keep adapters thin and pin the pivot
  format (OTel gen_ai) so most maintenance is one shared path.
- **Scope creep.** Ship the OTLP path + Langfuse first; resist adding five
  adapters before one is proven end-to-end.

## 7. Sequencing

1. `OTLPImporter` reading OTel gen_ai spans → `TraceDataset` (reuses
   `to_trace_record()`; smallest possible first cut).
2. `LangfuseAdapter` + the extractor/label-space-inference layer (§5.1) — the
   real design work.
3. `tracer import` CLI wiring + `[import]` extra + docs: rewrite the "bootstrap
   from scratch" section of `cold_start.md` around warm start.
4. Second adapter (LangSmith or Phoenix) to validate the interface generalizes.

## 8. Open questions

- **Extractor UX:** is JSONPath + callable enough, or do we need an interactive
  "here are 10 sample outputs, pick the label field" onboarding step (mirrors the
  agentic `cloud onboard`)?
- **Label-space confirmation:** auto-fit, or always require the user to approve
  the inferred taxonomy first?
- **Ground truth:** some platforms store eval scores / human annotations
  alongside generations — should the importer populate `TraceRecord.ground_truth`
  from those when present (enables the paper's ground-truth-vs-teacher analysis)?
- **Incremental import:** should re-running `import` be idempotent (dedup by
  `trace_id`) so it can top up an existing `all_traces.jsonl` between `watch`
  runs?
