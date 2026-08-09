# Hardware Design Space

Hardware exploration begins with a typed, versioned candidate definition. A candidate is not just a device name or peak-FLOP number; it is a composable blueprint of resources, topology, constraints, and implementation capabilities that can be mapped, simulated, compared, and revised.

!!! note "Design status"
    This page defines the accepted product model. The current repository has `SystemProfile` evidence but not yet a complete public `ArchitectureBlueprint` schema or search API.

## Candidate blueprint

Conceptually, one candidate contains:

```text
ArchitectureBlueprint
├── compute_substrate
├── memory_hierarchy
├── on_chip_fabric
├── package_and_scale_up
├── scale_out_network
├── system_organization
├── capability_and_programming_model
├── physical_envelope
└── assumptions_and_revision
```

The blueprint owns architectural facts and variables. Measured/simulated latency belongs to evidence; a particular cluster allocation belongs to deployment; a workload mapping belongs to an experiment. Keeping these objects separate allows one architecture to be evaluated under several workloads, deployments, and evidence fidelities.

## Compute substrate

Compute parameters include engine kinds and counts, supported operations and precision, vector width, array dimensions, accumulator behavior, clock domains, issue/queue structure, local reduction support, sparsity, and dataflow constraints.

Peak throughput is a derived upper bound. Useful throughput depends on operation shape, tiling, utilization, data movement, synchronization, and concurrency. Blueprinting therefore connects compute capabilities to implementation candidates and evidence domains rather than assigning one universal efficiency.

## Memory hierarchy

Each memory level describes capacity, bandwidth, latency, ports, banks, alignment, addressability, consistency, and supported transfers. The hierarchy may span registers, local SRAM, shared cache, HBM, host memory, and remote memory.

The important exploration variables are not only “more capacity” and “more bandwidth.” Banking conflicts, reuse scope, placement, double buffering, collective staging, and lifetime pressure often determine whether compute engines can be fed. Memory planning must use actual buffer lifetimes from the mapped execution plan.

## Interconnect and communication

The communication model spans several scopes:

| Scope | Example variables |
|---|---|
| On-chip | NoC topology, routers, virtual channels, link widths, multicast/reduction support |
| Die/package | chiplet topology, die-to-die bandwidth/latency, coherence or message semantics |
| Scale-up | device topology, collective offload, link class, routing, oversubscription |
| Scale-out | NICs, switches, topology, rail mapping, congestion and failure domains |

Logical message bytes come from the workload mapping. Physical traffic and exposed time depend on collective algorithm, route, contention, and implementation. Network simulators therefore refine—not redefine—the logical communication facts.

## System organization

A system candidate composes chips into packages, nodes, racks, and clusters. It specifies multiplicity, affinity, topology, host involvement, storage or offload paths, power domains, and resource reservations.

Architecture and deployment remain distinct. “Eight identical devices connected by topology X” can be an architecture template; “these eight serial-numbered devices with current reservations” is a deployment. Exploration normally evaluates templates before binding a real deployment.

## Physical and economic envelope

Architecture decisions eventually face area, power, thermal, packaging, yield, and cost constraints. These metrics require their own versioned providers, just like latency:

- analytical area and energy models for early pruning;
- component or floorplan models for refinement;
- power/thermal simulation for selected candidates;
- BOM, packaging, and deployment-cost models for system comparison.

An absent model remains unknown. Blueprinting must not infer that a design is Pareto-optimal by omitting an inconvenient objective.

## Capability and programmability

The blueprint declares supported primitives, layouts, synchronization, queues, memory operations, collective capabilities, runtime/ABI constraints, and extensibility points. These facts determine whether a workload mapping is legal and how much software specialization it requires.

LPU, GPU, and other accelerators may expose materially different programming models. The portable workload and mapping intent remain shared; target plugins interpret capabilities only at the architecture-binding boundary.

## Design variables and constraints

Each field is classified as:

- fixed fact;
- enumerable or bounded design variable;
- derived quantity;
- hard constraint;
- soft preference;
- unresolved value requiring a provider or later binding.

Constraints include divisibility, topology feasibility, capacity, bandwidth, power, area, implementation legality, and experiment budgets. Conditional variables prevent meaningless combinations—for example, router-buffer depth exists only when the selected fabric uses buffered routers.

## Hierarchical search

The design space is too large for flat Cartesian enumeration. Search proceeds hierarchically:

1. use exact work and capacity bounds to reject impossible candidates;
2. use analytical models for coarse architecture pruning;
3. construct legal mappings for survivors;
4. use schedule-aware estimates and network/hardware simulation selectively;
5. retain Pareto and uncertainty-relevant candidates;
6. spend detailed simulation or measurement budget near decision boundaries.

Equality saturation may generate equivalent mappings or local implementations, but it is one candidate-generation technique inside this hierarchy—not the product's organizing abstraction.

## Reproducible candidate identity

Every result records the blueprint digest, variable assignments, schema revision, generator/search revision, workload suite, mapping digest, deployment context, evidence snapshot, simulator revisions, objectives, constraints, seed, and budget.

Changing any material assumption creates a new candidate or experiment identity. This is what turns architectural exploration from informal spreadsheet tuning into a reproducible engineering process.

See the [hardware architecture model](../modeling/hardware.md) for schema ownership and the [exploration workflow](workflow.md) for experiment construction.
