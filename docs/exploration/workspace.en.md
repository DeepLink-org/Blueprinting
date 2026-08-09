# Exploration Workspace

The exploration workspace is the interactive surface of the Blueprinting workbench: a single-page NiceGUI application backed by the framework-neutral `BlueprintingService`. It is a presentation layer over the analysis stack — it introduces no new semantic layer and never becomes an alternative source of workload truth.

!!! note "Terminology"
    “Workspace” appears in two distinct senses in the documentation: this page describes the workbench's **exploration workspace** (the interactive views for four modes); `PortablePlanIR` separately defines the `WORKSPACE` buffer role, a capacity constraint for abstract working memory, in [Planning and Execution IR](../design/ir/planning-execution.md).

## What the workspace is

The workspace renders one of four mode views, all sharing the same configuration surface. Analysis runs outside the UI event loop; a result is shared across views and marked stale after configuration changes; failed candidates remain visible as structured diagnostics instead of vanishing silently. It consumes the framework-neutral services in `application/` and shares the same typed contracts as the CLI.

## Shared configuration surface

Point and batch modes share one configuration dialog: model/execution presets (from the retained JSON presets in `data/`), TP/PP/DP topology, calibration mode (system-evidence curve vs theoretical peak baseline), and candidate bounds. Quick controls and the full configuration stay in sync.

## Workspace modes

### Point lens

Focuses on one case: runs the `ModelIR -> DistributedTaskIR -> PortablePlanIR` derivation, resolves evidence estimates, and shows task-level contributions, resource constraints, and evidence boundaries. A failed derivation keeps a diagnostic with lineage instead of fabricating a performance number.

### Batch lens

Treats a set of TP/PP/DP cases as a whole: each case derives independently, exposing distributions, upper/lower bounds, and the feasible boundary while shrinking the candidate space. Failed items retain status and diagnostics and participate in the distribution statistics.

### Evidence lab

A read-only evidence catalog: pinned Vidur Phi-2/A100 records with exact selectors. GEMM primitives compare measured versus analytical roofline on identical workload facts; the measured series shows exact sample points without claiming interpolation; attention and other operations appear in the coverage catalog only.

### Numeric lens

Hosts the [Floating-Point Numerical Analysis](../design/numerical-analysis.md) panel: formats, encoding, dynamic range, and operation boundaries. It is usable independently of workload analysis and does not require running a derivation first.

## Workspace invariants

- no eager analysis on load;
- analysis executes outside the UI event loop (`io_bound`), keeping the interface responsive;
- a single result is shared across views and marked stale after configuration changes;
- sweep candidates retain per-case status and failure diagnostics;
- evidence is read-only: the lab never writes back to evidence or canonical IR;
- the optional portable dependency Chrome Trace export is explicitly marked `executable=false` and is not a `TimelineBundle`.

## Architecture relationship

The workbench consumes `application/` services and presents canonical derivation audits, cost resolution, and evidence catalogs. It does not construct canonical plans itself; every analysis entry point shares the typed contracts and session bindings of the CLI.

## Current boundary

The workspace is an analysis and audit surface, not a simulator. It has no first-class `ArchitectureBlueprint` editing, no design-space search, and no discrete-event simulation; those remain planned product slices (see [Implementation Status](../project/status.md) and the [exploration workflow](workflow.md)).
