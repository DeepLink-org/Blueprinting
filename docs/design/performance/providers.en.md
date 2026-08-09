# Cost Providers and Performance-Data Imports

Blueprinting now has a runnable task-cost seam between portable workload facts and plan-level composition. The implementation is intentionally narrower than a complete performance service: it resolves latency for one operator or communication task, preserves source identity, and refuses ambiguous evidence. It does not yet interpolate arbitrary shapes, model contention, or replace schedule simulation.

## Why this boundary exists

`PortablePlanIR` owns work—operations, bytes, messages, dependencies, and semantic shape. It must not own a duration measured on one runtime or one GPU. A cost query combines that immutable work with late-bound architecture, runtime, implementation, and deployment context:

```text
PortablePlanIR task + late-bound CostQueryContext
  -> CostQuery
  -> CostResolver(ordered providers)
       1. PerformanceDatabaseProvider
       2. another measured/simulated provider
       3. RooflineCostProvider
  -> CostEstimate + provider attempts + revisions
  -> inference cost view / future schedule simulation
```

The resolver selects one provider. It does not multiply corrections or average unrelated sources. A miss may proceed to the next provider; ambiguous or internally conflicting evidence is an error and cannot be hidden by a fallback.

## Normalized contracts

`CostQuery` has first-class fields for:

- subject (`operator` or `communication`), operation, hardware, and datatype;
- operations, read/write bytes, message bytes, participants, network tier, and engine;
- hardware, implementation, runtime, topology, and power-mode revisions;
- operation-specific dimensions such as `m/n/k`, batch, tokens, context, heads, or backend.

Every field participates in the canonical query digest. Empty optional identity fields mean “unknown”; they do not match a record that explicitly requires a runtime or kernel.

`CostEstimate` reports latency together with method, match kind, provider/source revisions, raw record IDs, validity selector, uncertainty, components, and assumptions. The current methods are measured, simulated, analytical, calibrated, and vendor-model; the current database provider emits only exact-selector results.

## Roofline provider

`RooflineCostProvider` wraps a versioned `SystemProfile`. For a local operator it computes:

```text
compute_time = operations / effective_engine_throughput
memory_time  = (read_bytes + write_bytes) / effective_memory_bandwidth
roofline     = max(compute_time, memory_time)
```

Peak-only and system-evidence efficiency modes are explicit. `processing_mode="roofline"` uses the max bound; `"no_overlap"` uses the explicit serialized sum; `"profile"` adopts the legacy profile setting. The estimate exposes both components, arithmetic intensity, and the selected bottleneck.

Communication queries use the selected `NetworkProfile`: collective volume rule, participant count, bandwidth, efficiency, and launch latency remain visible inputs. This is an analytical collective model, not a claim of cycle-accurate network simulation.

## Immutable performance database

A `PerformanceRecord` contains a latency sample, a declared selector, and `EvidenceProvenance`:

```text
source + source revision
importer revision + source-file digest
method + raw record ID
selector + optional source metadata
```

`PerformanceDatabase` is canonically serializable and content-addressed. `PerformanceDatabaseProvider` finds all records whose declared selector is an exact subset match of the query context, then selects the most-specific selector. Repeated samples from the same provenance group are aggregated by median and report population deviation and range.

Provider construction builds an immutable index by core identity, selector schema, and typed selector values. Query resolution therefore does not linearly scan a full imported corpus; the database digest is computed once and cached as provider identity.

Two equally specific but different provenance groups are ambiguous. The provider rejects them instead of choosing the fastest row or silently blending revisions. Interpolation is deliberately absent from this first slice; it should later be a named provider with an explicit validity domain and extrapolation distance.

## Importing a simulator table

`SimulatorPerformanceImporter` accepts CSV, JSON/JSONL, and—when the `performance-data` extra is installed—Parquet. A `TabularImportSpec` declares every mapping and the latency unit:

```python
from blueprinting.analysis import (
    CostSubject,
    EstimateMethod,
    LatencyUnit,
    SimulatorPerformanceImporter,
    TabularImportSpec,
)
from blueprinting.synthesizer.frozen import FrozenDict

spec = TabularImportSpec(
    name="noc-sim-r7",
    subject=CostSubject.COMMUNICATION,
    source="our-network-simulator",
    source_revision="git:4e5c...",
    method=EstimateMethod.SIMULATED,
    latency_column="latency_us",
    latency_unit=LatencyUnit.MICROSECONDS,
    operation_column="collective",
    hardware="lpu-candidate-17",
    datatype_column="dtype",
    selector_columns=FrozenDict({
        "message_bytes": "bytes",
        "participants": "ranks",
    }),
    selector_types=FrozenDict({
        "message_bytes": "int",
        "participants": "int",
    }),
    record_id_column="run_id",
)
database = SimulatorPerformanceImporter.from_file("collectives.csv", spec)
```

The importer does not infer units or coerce selector columns heuristically. Invalid, empty, non-finite, or incorrectly typed values fail ingestion.

## Importing Vidur profiles

`VidurProfileImporter.from_csv()` understands Vidur's attention and MLP profile schemas and converts supported timing columns into normalized records. Attention records preserve phase, batch, context/cache semantics, TP degree, backend, block size, and model shape. Compute records preserve token count and model/parallel dimensions.

This does **not** change the oracle boundary:

- `VidurProfileBaseline.lookup()` remains a post-hoc comparison oracle and is not a `CostProvider`;
- `VidurProfileImporter` is an explicit user action that creates a new Blueprinting-owned evidence database;
- only a `PerformanceDatabaseProvider` deliberately installed in a resolver may affect costing.

The distinction prevents a validation baseline from leaking into lowering while still allowing independently reviewed profile data to become admissible evidence.

## Importing NVIDIA AIConfigurator data

The official [AIConfigurator repository](https://github.com/ai-dynamo/aiconfigurator) stores component performance in heterogeneous operation-family Parquet tables. `AIConfiguratorPerformanceImporter` currently has strict adapters for:

| Upstream table | Normalized subject/operation | Required shape context |
|---|---|---|
| `gemm_perf` | operator / `gemm` | `m`, `n`, `k`, dtype |
| `context_attention_perf` | operator / `attention_core` prefill | batch, input length, local heads, head size, KV dtype |
| `generation_attention_perf` | operator / `attention_core` decode | batch, `isl + step`, local heads, head size, KV dtype |
| `custom_allreduce_perf` | communication / `all_reduce` | message bytes, GPU count, backend |

AIConfigurator latency is normalized from milliseconds to seconds. Framework, framework version, and kernel source become required runtime/implementation selectors; device and upstream operation identity remain provenance metadata. Therefore an AIConfigurator row cannot match a hardware-only query whose runtime/kernel is unknown.

```python
database = AIConfiguratorPerformanceImporter.from_file(
    "gemm_perf.parquet",
    hardware_name="h100-sxm",
    source_revision="8fc57cf...",
)
```

AIConfigurator's final `best_config_topn.csv` and Pareto outputs describe serving configurations, not atomic operator evidence. They are intentionally not imported as task costs. A future plan-level comparison adapter may consume them without flattening TTFT/TPOT into kernel latency.

## Using the resolver in inference costing

Static inference can use the new resolver while the legacy `InferenceCostProvider` seam remains compatible:

```python
resolver = CostResolver((
    PerformanceDatabaseProvider(database),
    RooflineCostProvider(hardware),
))

context = CostQueryContext(
    runtime="vllm",
    runtime_revision="0.24.0",
    implementations=FrozenDict({
        "attention_pre_projection": "torch.nn.functional.linear",
        "all_reduce": "vLLM_custom_graph",
    }),
    operation_dimensions=FrozenDict({
        "all_reduce": FrozenDict({"backend": "vllm_graph"}),
    }),
)

estimate = estimate_inference_phase(
    plan,
    hardware,
    cost_resolver=resolver,
    cost_context=context,
)
```

Inference task queries are derived from canonical `PlanTask.workload`; GEMM dimensions and local attention-head dimensions are derived from model and TP facts. Tensor-parallel collectives and pipeline P2P use the same resolver. All tasks must be covered by an installed provider—normally an exact database followed by roofline—so an unknown task never becomes zero.

## Implemented boundary and next steps

Implemented now:

- normalized immutable query/estimate/support/resolution contracts;
- deterministic ordered resolution with an auditable attempt trace;
- analytical roofline and collective fallback;
- immutable exact-selector database with repeated-sample aggregation;
- generic simulator/profiler table ingestion;
- explicit Vidur and four-family AIConfigurator ingestion;
- inference task and pipeline integration with regression tests.

Still missing:

- calibrated interpolation/extrapolation providers and confidence policy;
- append-only raw-sample/environment-manifest storage beyond the portable snapshot;
- training-path migration through resolver equivalence tests;
- first-class KV-head/GQA and runtime legalization context in target binding;
- contention, overlap, queueing, and plan-level discrete-event simulation;
- energy/power metrics as normalized planner objectives;
- observation ingestion and calibration revisions.

These omissions are important: task latency resolution is an evidence layer, not a completed hardware or serving simulator.
