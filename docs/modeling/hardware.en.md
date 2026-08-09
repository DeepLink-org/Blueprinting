# Hardware Architecture Model

The hardware model is the semantic description of a candidate architecture. It defines what resources and capabilities exist and how they connect. Performance evidence estimates how those resources behave; deployment identifies concrete instances. These concerns must remain separate for design-space exploration to be meaningful.

!!! note "Design status"
    The current `SystemProfile` implements a limited evidence profile for compute, memory, and networks. The hierarchical architecture schema described here is the target design and is not yet connected end to end.

## Architecture, deployment, and evidence

| Object | Owns | Must not own |
|---|---|---|
| `ArchitectureBlueprint` | resource hierarchy, capabilities, topology, design variables, physical constraints | benchmark observations or concrete device allocation |
| `DeploymentProfile` | concrete devices, links, reservations, environment and runtime revision | portable workload semantics or architecture search variables |
| `EvidenceSnapshot` | measurements, simulator results, calibrated models, uncertainty and provenance | architectural truth or workload meaning |

One blueprint can be tested in several deployments and evidence snapshots. One evidence database can serve several compatible blueprints. Changing an efficiency curve must not change the architecture digest.

## Hierarchical composition

The model composes reusable typed components:

```text
cluster
  -> nodes and scale-out links
  -> packages and scale-up links
  -> dies/chiplets and die-to-die fabric
  -> tiles
  -> compute engines + local memories + on-chip network endpoints
```

Every component has a stable identity, multiplicity, clock/power domain, parent scope, ports, and typed connections. Templates define a family; bound parameter values define one candidate blueprint.

## Compute model

A compute engine declares supported operation families, datatype and accumulation rules, shape/layout limits, concurrency, local storage access, issue semantics, and synchronization capabilities. Examples include matrix arrays, vector units, scalar/control cores, reduction engines, DMA engines, and collective accelerators.

The architecture model records capacity and legality. Throughput/latency curves remain evidence keyed by operation, shape, implementation, engine revision, and context. This avoids embedding one optimistic utilization factor into the engine definition.

## Memory model

Memory spaces declare:

- scope and visibility;
- capacity and allocatable reservations;
- banks, ports, channels, and address granularity;
- supported transfers, multicast, reduction, and coherence semantics;
- connectivity to engines and other memory levels;
- alignment, layout, and allocation constraints.

Bandwidth and latency evidence can vary with transfer size, access pattern, bank mapping, occupancy, and contention. Buffer placement and lifetime are decisions of the mapped plan, not intrinsic fields of the memory component.

## Interconnect model

An interconnect consists of endpoints, links, routers/switches, topology, routing capability, arbitration, buffers, and supported communication operations. The same abstraction spans NoC, chiplet links, scale-up fabrics, and scale-out networks while allowing scope-specific extensions.

Simple analytical providers may expose latency/bandwidth and collective-volume curves. Detailed simulators consume the same topology identity and return route/link events, congestion, and uncertainty. Logical workload messages remain upstream facts.

## Capability and legality model

Capabilities answer whether and how a workload primitive can execute. They include operation semantics, precision, layout, shape range, memory accessibility, queue/event behavior, collective support, runtime requirements, and ABI constraints.

Legality is deterministic for a blueprint and mapping request. Missing support produces a diagnostic with the source task, violated rule, and considered alternatives. It is never converted into an infinite or zero cost hidden inside estimation.

## Physical metrics

Area, energy, power, thermal, yield, and cost are derived through providers attached to component and system identities. Early providers may be analytical; selected candidates may use floorplan, RTL, power, thermal, or packaging tools.

The model distinguishes quantities that are additive, peak, averaged, or state-dependent. A system power envelope, for example, is not the sum of unrelated peak component numbers unless the execution plan can activate them simultaneously.

## Fidelity levels

The same architecture identity supports several evaluation levels:

1. structural feasibility and exact capacity bounds;
2. analytical throughput/bandwidth/energy models;
3. empirical or surrogate component models;
4. event-level resource simulation;
5. detailed network, hardware, power, or thermal simulation;
6. prototype or silicon measurement.

A result records its fidelity and validity domain. Higher fidelity refines evidence; it does not silently change component semantics.

## Extension and target plugins

Common components cover shared concepts, while target plugins provide architecture-specific capabilities, legality, implementation catalogs, simulator adapters, and optional MachineIR emission. An LPU plugin may expose specialized dataflow or memory operations without forcing GPU candidates into the same instruction model.

The core requirement is comparable experiment semantics: shared workload facts, explicit architecture binding, normalized evidence results, resource-correlated traces, and stable provenance.

## Versioning and verification

An architecture blueprint verifies component references, topology connectivity, scope, multiplicity, clock/power-domain rules, conditional variables, unit consistency, and constraint satisfiability. Its canonical digest includes all semantic fields but excludes derived estimates.

Schema migration is explicit. Published experiments retain the original blueprint and provider revisions so an architectural conclusion can be reproduced after the model evolves.

## Current implementation gap

`SystemProfile` currently supplies matrix/vector throughput curves, memory capacity/bandwidth curves, network tiers, and collective models used by the Calculon calibration. It does not yet model component hierarchy, NoC, queues, power/area/cost, architecture variables, or a general target capability graph.

The first migration step is to wrap `SystemProfile` as evidence for a minimal virtual `ArchitectureBlueprint`, preserving existing results while introducing the separation above. See the [roadmap](../project/roadmap.md).
