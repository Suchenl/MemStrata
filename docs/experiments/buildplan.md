# MemStrata submission-week build plan

## Non-negotiable priority

This week has one delivery order: freeze paper-facing gold, run reproducible
experiments on that gold, then fill and compile the paper. No new features are
accepted until a listed delivery gate is complete. The repo-wide current-focus
pointer is [`assets/docs/hot.md`](../../../../assets/docs/hot.md).

## Milestones

| Deadline | Deliverable | Evidence gate |
|---|---|---|
| July 22 | All paper-facing samples reach S7 | `human_reviewed=true`; S3/S4/S6 decisions resolved; strict lint has zero errors; generated readiness report records every catalog row |
| July 22 | Main and ablation experiments finish | Each paper claim maps to a tracked driver, `results.json`, `results.md`, and a green or explicitly negative row in [`experiments/REGISTRY.md`](../../../../experiments/REGISTRY.md) |
| Submission day | Paper is evidence-complete | Tables and figures only cite frozen-run results; claims match evidence; PDF compiles cleanly |

## Gold gate

The authoritative S1–S7 annotation protocol is
[`benchmark/annotation_pipeline.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/benchmark/annotation_pipeline.md)
(scoring/replay is in [`benchmark/scoring.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/benchmark/scoring.md)).
The freeze/review gates are in [`benchmark/annotation_pipeline.md`](https://github.com/Suchenl/VMem-Bench/blob/main/docs/benchmark/annotation_pipeline.md);
the three-role mental model is in [`overview/README.md`](../overview/README.md). A sample is not
finished because an automation run exited successfully: it must pass the human-review
and strict-freeze gates.

Generate the deterministic status report from an explicit catalog:

```bash
PYTHONPATH=benchmarks/VMem-Bench/src python3 \
  benchmarks/VMem-Bench/scripts/vmem_bench/maintenance/report_gold_readiness.py \
  --catalog <catalog.jsonl> --out <readiness.json>
```

The report records stage state, S4/S6 approval, S7 strict-lint state and gold
manifest presence per catalog row. It never turns a candidate into gold.

## Experiment-to-paper gate

The paper's contribution/baseline/table contract is in
[`docs/paper/paper_organization.md`](../paper/paper_organization.md).
Diagnostics (empty/full/oracle) may explain limits but are not substitutes for
the main baseline table. A run counts only when its driver and outputs are
registered in [`experiments/REGISTRY.md`](../../../../experiments/REGISTRY.md).

## Daily operating rule

1. Remove the earliest gate blocker, beginning with S6/S7 for BBB.
2. Record the deterministic gold readiness report and experiment result after
   every completed batch.
3. If a result weakens a claim, downgrade the paper text instead of hiding the
   result. Continue until all three submission gates above are complete.
