# MemStrata Documentation Index (single entry point)

> The project is split into two self-contained repositories: **method
> documentation lives here in this repository's `docs/`** (this index), while
> **benchmark / baseline / bench-side experiment documentation lives in the
> [VMem-Bench repository](https://github.com/Suchenl/VMem-Bench)**, organized into
> per-role directories.
> Goal: eliminate semantic drift, so that "method / benchmark / baseline /
> experiment / paper" each have a single authoritative home.
> Entries below that point to the benchmark side use links to the VMem-Bench
> repository so they also work after this repository is cloned independently.
> Before adding, moving, or renaming a document, read this index first; **if a
> near-equivalent document already exists, edit it instead of creating a new one**.
>
> Terminology has [`glossary.md`](glossary.md) as its **single source of truth**;
> do not define terms separately in other documents.

## The three roles at a glance (stop conflating them)

This project has **three kinds of things**, and the documentation directories map
one-to-one onto them:

| Role | What it is | Code package | Docs directory |
|---|---|---|---|
| **Method / SUT** | `memstrata` — **the memory-management + context-composition approach we propose**; it is **the thing being evaluated** | `src/memstrata/` | [`method/`](method/) (Chinese) |
| **Benchmark / Bench** | `vmem_bench` — the annotation pipeline that **evaluates this method** → frozen gold → deterministic scoring | `src/vmem_bench/` | [`benchmark/`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/) |
| **Baseline / Control** | **Other systems** compared against the SUT on the **same benchmark** (causal / retrieval / diagnostic) | `baselines/`, `baseline_adapters/` | [`baselines/`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/baselines/) |

> ⚠️ **`memstrata` is the SUT (the method under test), not one of the baselines.**
> `method/` and `baselines/` are always written separately.

## Directory structure

| Directory | Contents | When to read |
|---|---|---|
| [`overview/`](overview/) | Mental model of **how each of the three roles operates** (this index only tells you "where to find things"; overview covers "how things work") | To build a correct understanding within ten minutes |
| [`method/`](method/) (Chinese) | Design philosophy and implementation specification for **the SUT (`memstrata`)** | Before editing `src/memstrata/` |
| [`benchmark/`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/) | Authoritative documents for the **benchmark (`vmem_bench`)** protocol / contracts / annotation / scoring | Before editing `src/vmem_bench/` or scoring |
| [`baselines/`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/baselines/) | Baseline strategies, Track A implementation, **fairness decisions** | Before running / editing / evaluating a baseline |
| [`experiments/`](experiments/) (Chinese) | Experiment plans, ablations, **fairness experiment plan** | Before planning / reproducing experiments |
| Paper materials | The paper source is maintained separately from this method package; use [`REPRODUCE.md`](../REPRODUCE.md) for frozen experimental reproduction | Before writing / editing the paper |
| [`operations/`](operations/) | **Single authoritative list of weights / environments / run requirements** (model paths, conda envs, env variables, calibration, release self-containment checklist) | Before swapping weights / configuring environments / running GPU jobs / releasing |
| [`glossary.md`](glossary.md) | Unified terminology table (single source of truth) | Any time you are unsure about wording |

## Per-directory inventory

### method/ (SUT = the authoritative design for `src/memstrata/`; **not a baseline**)
- [`philosophy.md`](method/philosophy.md) (Chinese): **the highest-level charter** — the six library-quality axioms, WHO before WHERE, the boundary between name anchoring vs embedding, conservative rejection/isolation.
- [`design.md`](method/design.md) (Chinese): the implementation specification aligned with the paper's four-step loop (including the D6 zero-import boundary, planner fallback constraints, visual strata, Track A adapter).
- [`generator_wiring.md`](method/generator_wiring.md) (Chinese): the video-generator backend inventory and wiring.

### benchmark/ (Bench = the authoritative `vmem_bench` protocol; maintained in the public [VMem-Bench benchmark docs](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/))
- [`running_eval.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/running_eval.md): the **end-to-end run manual** — how to take a film from frozen gold all the way to a score (Stage 1 produces context → Stage 2 VLM scores). Read this first if you want to "run one film".
- [`design_principles.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/design_principles.md): the design principles for this benchmark (including the general benchmark-methodology charter).
- [`schemas_and_contracts.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/schemas_and_contracts.md): data contracts and metric definitions (**the authoritative schema source**; for the headline metric see `scoring.md`).
- [`annotation_pipeline.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/annotation_pipeline.md): the **stage-level authority** for the annotation pipeline (S1–S7 produces frozen gold).
- [`annotation_tracking_internals.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/annotation_tracking_internals.md): the **internal mechanics** of track-first tracking / re-ID / identity resolution (referenced by the `pipeline_track_first` code).
- [`scoring.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/scoring.md): the **scoring authority** — VisualFidelity headline, multi-embedder routing, deterministic replay, LSMDC-specific metrics.
- [`crop_contract.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/crop_contract.md): the overall crop principles and attribute-field contract shared by Bench and SUT.
- [`services_and_time.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/services_and_time.md): resident model services, GPU placement, time metadata.
- [`dashboard_and_review.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/dashboard_and_review.md): the SSE monitoring + human-review UI task spec and the human-in-the-loop review strategy.
- [`staged_pipeline_plan.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/staged_pipeline_plan.md): the S2/S3/S4 staged pipeline optimization plan.
- [`pitfalls.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/pitfalls.md): incident-and-fix records from running (institutional memory).
- [`references.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/benchmark/references.md): references and the 2026 landscape.

### baselines/ (other systems compared against the SUT on the **same benchmark**; maintained in the public [VMem-Bench baseline docs](https://github.com/Suchenl/VMem-Bench/tree/main/docs/baselines/))
- [`fairness_decisions.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/baselines/fairness_decisions.md): the **final decisions on fair baseline comparison** (frozen 2026-07-22, currently authoritative).
- [`track_a.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/baselines/track_a.md): Track A = the implementation of real retrieval / memory operating on GT visuals (including each baseline's weight source and placement).
- [`strategy.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/baselines/strategy.md): baseline strategy and the historical selection record (selection is governed by `fairness_decisions.md`).
- [`external_baseline_audit.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/baselines/external_baseline_audit.md): the historical verification table for external baselines (superseded by `fairness_decisions.md`).
- [`hook_recipes.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/baselines/hook_recipes.md): recipes for instrumenting external systems to export evidence.

### experiments/
- [`buildplan.md`](experiments/buildplan.md) (Chinese): the paper-facing delivery order and acceptance gates.
- [`fairness_experiment_plan.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/experiments/fairness_experiment_plan.md): the **fairness experiment plan** (name-anchored / description-only input sets, multi-embedder, k sweep, causal coverage). **(Bench-side document)**
- [`generator_in_the_loop_eval_plan.md`](https://github.com/Suchenl/VMem-Bench/tree/main/docs/experiments/generator_in_the_loop_eval_plan.md): the **generator-in-the-loop evaluation design (discussion draft)** — long-range consistency degradation curves, sample size, and the division of labor with the Track A main table. **(Bench-side document)**
- [`open_source_movie_track_decomposition.md`](experiments/open_source_movie_track_decomposition.md) (Chinese), [`ablation_study/`](experiments/ablation_study/) (Chinese).

### Paper
The manuscript is maintained separately from this code repository. For the
frozen experiment protocol and reproduction entry points, use
[`REPRODUCE.md`](../REPRODUCE.md) and the public
[VMem-Bench documentation](https://github.com/Suchenl/VMem-Bench).

## Not inside docs/ (deliberately left in place; only registered here as pointers)

These are **runtime code assets, package-entry contracts, or build artifacts**;
moving them would break functionality or semantics, so they stay where they are:

- `../AGENTS.md`: the hard architectural-boundary declaration (`memstrata` ↔ `vmem_bench` zero mutual imports, zero external dependencies).
- `src/*/README.md`, `scripts/*/README.md`: package/directory entry contracts (Design docs always point back to this `docs/`).
- `src/vmem_bench/annotation/pipeline/stages/**/*.md`: prompt templates and review checklists **read at runtime** by the annotation pipeline.
- `assets/paper/MemStrata/sections/`, `_figures/`, `notation.md`: LaTeX build artifacts for the paper.
- `data/_runs/*/results.md`, `data/_vlm_rerun_kit_*/`: experiment artifacts / temporary annotation data, not knowledge documents.
- `baselines/{Scripted,Causal}/**`: external vendored source checkouts (with their own READMEs), not knowledge documents of this project.
