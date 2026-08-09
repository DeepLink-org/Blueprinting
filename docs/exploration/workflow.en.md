# Exploration Workflow

A hardware exploration is a reproducible experiment, not a sequence of ad-hoc calculator edits. The workflow separates the question, workloads, architecture variables, mappings, evidence fidelity, simulation, and decision criteria so that conclusions remain explainable and revisable.

## Frame the decision

Begin with a decision that could change a hardware blueprint:

- Is additional matrix throughput useful, or is memory/communication already limiting?
- Should scarce area be spent on SRAM, HBM interfaces, NoC bandwidth, or compute tiles?
- Which scale-up topology sustains tensor/pipeline parallel workloads?
- How does the best architecture change between training, prefill, and decode?

Declare objectives, hard constraints, uncertainty tolerance, evaluation budget, and the decisions that are out of scope. “Make it faster” is not an experiment specification.

## Define the workload suite

A workload suite captures representative model families, shapes, sequence lengths, batch regimes, training/inference modes, precision, parallel strategies, and scenario weights. It includes stress cases and anticipated future workloads, not only one convenient benchmark.

The workload model derives exact operations, data movement, communication, dependency, and lifetime facts independently of the candidate architecture. This is the controlled baseline shared by every candidate.

## Generate candidate blueprints

Define fixed facts, design variables, conditional variables, and constraints for compute, memory, interconnect, system organization, and physical envelope. Record the generator revision and why each variable range is credible.

Candidate generation may combine enumeration, parameterized templates, domain rules, equality-based mapping alternatives, Bayesian optimization, or expert proposals. All candidates become ordinary versioned blueprints before evaluation.

## Construct legal mappings

For each surviving blueprint, map the workload into implementations, shards, placement, queues, communication routes, synchronization, and buffers. This stage answers feasibility before performance:

- does the target support every required primitive and datatype?
- do tensor/layout/alignment constraints hold?
- do buffers fit every memory level under possible overlap?
- are topology, queue, and synchronization constraints legal?

The same portable workload may yield different concrete plans on GPU, LPU, or experimental architectures. Invalid candidates receive source-linked diagnostics, not invented performance numbers.

## Select evidence and fidelity

Choose the cheapest evidence that can resolve the current decision:

| Fidelity | Typical use |
|---|---|
| Exact bounds | capacity, operation/byte volume, dependency depth |
| Analytical model | broad early pruning and sensitivity |
| Performance database/surrogate | known kernels, transfers, and collectives |
| Network simulator | topology, routing, congestion, collective alternatives |
| Hardware simulator | engine, memory-system, pipeline, power, or cycle detail |
| Measurement | calibration and final validation on prototypes or existing devices |

Every fallback, interpolation, extrapolation, and uncertainty bound is recorded. High fidelity is spent near decision boundaries rather than uniformly across the entire space.

## Simulate the mapped plan

Simulation consumes the verified architecture-bound command/resource plan. It computes readiness from dependencies, queues, synchronization, and resource capacity—not predicted timestamps. The trace records utilization, contention, memory high-water marks, critical paths, queue delay, and metric provenance.

Task costs and network behavior may come from different providers, but the simulator composes them through one resource model. This prevents each subsystem from assuming incompatible overlap or scheduling semantics.

## Compare and explain

The result analysis performs:

- feasibility filtering and constraint diagnostics;
- Pareto comparison across declared objectives;
- bottleneck attribution by resource and workload phase;
- sensitivity and what-if analysis over architecture variables;
- robustness analysis across workload scenarios and uncertainty;
- regression comparison against previous blueprint/evidence revisions.

A useful recommendation states both the winning candidate and the conditions under which that conclusion changes.

## Validate and calibrate

Compare predicted events and counters with compatible simulator, prototype, GPU/LPU, or silicon observations. Correlate residuals to workload tasks, concrete commands, implementations, and resources before fitting a new model.

Calibration creates a new immutable evidence revision with its training set, validity domain, and validation error. Historical experiments remain reproducible; adopting the new revision triggers explicit reevaluation rather than mutating old reports.

## Publish the experiment bundle

A complete experiment bundle contains:

```text
question and hypothesis
workload-suite digest
candidate blueprint definitions and generator revision
mapping/concrete-plan digests
evidence and simulator revisions
objectives, constraints, uncertainty policy, seed, and budget
results, traces, diagnostics, and rejected candidates
recommendation and claim boundary
```

This bundle is the durable Blueprinting product. Optional runtime configuration or program artifacts are attached to the selected blueprint when the target backend exists.

## Example progression

An LPU exploration might begin with analytical sweeps over matrix-engine count, SRAM capacity, HBM bandwidth, and scale-up links. Capacity and roofline bounds remove most candidates. Schedule-aware simulation then compares buffer reuse and compute/communication overlap. A network simulator resolves the finalists' collective behavior. Prototype observations calibrate the efficiency curves, and the revised Pareto set identifies which architecture should advance.

The exact algorithms may evolve; the experiment identity, semantic conservation, and evidence lineage remain fixed contracts.
