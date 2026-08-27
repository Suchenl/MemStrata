# Project primer: the three roles memstrata / vmem_bench / baselines

> Audience: anyone who wants to build a correct mental model within ten minutes.
> The whole project has **three roles that must not be conflated** — the method
> being evaluated (SUT), the benchmark that evaluates it, and the control systems
> run on that benchmark. Terminology has [`../glossary.md`](../glossary.md) as its
> single source of truth.

## Tell the roles apart in one sentence

> **`memstrata` is the method under evaluation (the SUT), not one of the
> baselines.** The baselines are the **other** systems placed on the same
> benchmark for comparison; the benchmark itself is neither the method nor a
> baseline.

This document only covers **how each of the three roles operates**; "which
document lives where" is the job of the [`../README.md`](../README.md) index
(including the role ↔ directory mapping) and is not repeated here. The two
packages **never import each other**; they interact only through pure JSON
contracts (`PromptPacket` / `ObservationPacket` / `ComposedContextRecord`). For
the hard boundary see [`../../AGENTS.md`](../../AGENTS.md).

## memstrata (SUT / our method)

`memstrata` replaces "the flat history that passive retrieval dumps out" with
"**intent-aligned entity composition**": on write, content is stratified per
entity (viewpoint / state / time / lifecycle, with embeddings used only for
deduplication); on read, the flow is `stable names in the prompt → keyword-match
to id → dictionary dereference → assemble a conditioning packet`, so the **read
path uses no VLM and no whole-bank similarity retrieval by default**. It is the
object this benchmark evaluates, and it produces a `ComposedContextRecord`
(named assets ⊕ role ⊕ lifecycle ⊕ instruction ⊕ forbidden). See
[`../method/design.md`](../method/design.md) (Chinese) and
[`../method/philosophy.md`](../method/philosophy.md) (Chinese) for details.

## vmem_bench (the benchmark)

The benchmark is a system that **constructs a frozen gold standard offline, then
scores by deterministic replay** — it is not an online system where a VLM judges
on the fly. It has three stages:

1. **Annotation**: candidate gold is constructed offline from a single-source,
   temporally continuous long video (detection/tracking/re-ID decide "who appears
   where and when", and the VLM/human reviewers only supply candidates and
   verdicts at controlled steps). The production pipeline is **S1–S7**.
2. **Freezing**: only gold that passes human review + strict lint is `freeze`d;
   `gold/*.json` + `embeddings.safetensors` are the sole scoring ground truth,
   and `present` / `forbidden` / scenario tags are never leaked to the SUT ahead
   of time.
3. **Scoring**: `load_gold` rejects gold that is unfrozen or whose hash does not
   match; `run_replay` processes each chunk in temporal order, first
   `PromptPacket → SUT ComposedContextRecord → score`, then emits an
   `ObservationPacket` (the SUT can build memory from the past, but cannot see
   the current gold before composing the current context). The headline metric is
   **VisualFidelity** (a multi-embedder routed by entity type).

For the causal ordering, metric definitions, and authoritative schema, see
[`benchmarks/VMem-Bench/docs/benchmark/`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/)
(`schemas_and_contracts.md` / `scoring.md` / `design_principles.md`).

## baselines (control systems)

The **other** systems compared against `memstrata` on the **same frozen gold, the
same harness, and the same pinned embedder**. The main quantitative table is the
**causal** systems (`helios / longlive_rag / memflow / iamflow / decmem`, matching
the paper's setting); the scripted / agentic systems (ViMax / MovieAgent /
VideoMemory / StoryMem / Memento / MM-StoryAgent) are non-causal and are removed
from the quantitative main table, appearing only as qualitative discussion in the
appendix. For the **current authority** on selection and fairness see
[`baselines/fairness_decisions.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/baselines/fairness_decisions.md);
for implementations see
[`baselines/track_a.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/baselines/track_a.md),
and for strategy/history see
[`baselines/strategy.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/baselines/strategy.md).

## Current status and where to look next

- **Gold status**: the BBB gold is **FROZEN**, with 52 chunks; the production
  annotation pipeline is **S1–S7**. For the authoritative account of how the
  annotations were produced see
  [`benchmark/annotation_pipeline.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/benchmark/annotation_pipeline.md).
- For the **delivery order and acceptance gates** (freeze → reproducible
  experiments → backfill the paper) see
  [`../experiments/buildplan.md`](../experiments/buildplan.md) (Chinese).

> Three lines for the mental model: ① the SUT is the method under evaluation, not
> a baseline; ② gold is frozen offline and scoring is a deterministic replay, not
> an online VLM judge; ③ for each chunk, scoring happens before the observation is
> emitted, guaranteeing the SUT cannot see the current ground truth before
> composing the current context.
