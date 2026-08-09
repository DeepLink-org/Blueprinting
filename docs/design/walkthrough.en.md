# Golden Formal Derivation Walkthrough

This walkthrough follows one row-parallel Transformer MLP fragment from architecture-independent work to a hardware-bound simulation plan. It is deliberately small enough to inspect, but it exercises semantic work, tensor parallelism, collective communication, portable mapping, architecture binding, scheduling, simulation, optional emission, and profiler lineage.

!!! note "Implementation status"
    The walkthrough through `PortablePlanIR` reflects the current Transformer slice. Later examples are illustrative pseudocode for the target design: `ConcretePlanIR` currently has only an experimental common schema, while typed target extensions, a scheduling producer, timing projection, and a MachineIR producer are not implemented.

## Source problem

Consider a row-parallel linear operation with symbolic token count (M), input width (N), output width (K), and tensor-parallel degree (P):

```text
Y[M, K] = X[M, N] × W[N, K]
```

Each rank owns (N/P) input columns, computes a partial output, and participates in an AllReduce. The exact local matrix work is `2 × M × (N/P) × K`; the logical collective payload is `M × K × element_bytes`.

## Stage 1: ModelIR

The frontend establishes model meaning without distribution or hardware:

```text
%y = model.linear %x, %w
    : tensor<MxNxbf16>, tensor<NxKxbf16>
      -> tensor<MxKxbf16>
```

Known after this stage:

- value and operation identity;
- shapes, dtype, state roles, and effects;
- the semantic relation between `x`, `w`, and `y`.

Still unknown: TP ownership, communication, recomputation, target kernels, placement, queues, buffers, and time.

The verifier checks definition/use, types, symbols, effects, and stable IDs. A profiler checkpoint can compare shape or numerical references but cannot attach target duration.

## Stage 2: DistributedTaskIR

The distribution pass binds a TP strategy and introduces logical ranks and communication:

```text
mesh @tp<P>

task @local_linear rank=%r {
  %partial = compute.matmul %x_shard[%r], %w_shard[%r]
  %y = collective.all_reduce %partial group=@tp reduction=add
}
```

The pass decides logical ownership and dependencies. It does not choose NCCL, a route, a physical device, or a collective algorithm.

The verifier proves that shards reconstruct the source tensors, every rank belongs to the mesh, collective participants match, the result has the original shape, and operation/message volumes are conserved.

## Stage 3: PortablePlanIR

Portable planning makes execution intent explicit while retaining target independence:

```text
portable.task @gemm {
  depends_on = []
  work = {operations = 2*M*(N/P)*K, reads = ..., writes = ...}
  requires = [matrix_engine(dtype=bf16)]
  concurrency_group = "compute"
}

portable.task @reduce {
  depends_on = [@gemm]
  work = {message_bytes = M*K*2}
  requires = [collective(kind=all_reduce, reduction=add)]
  concurrency_group = "network"
}
```

Abstract boundary buffers carry size, alignment, role, and lifetime constraints. An implementation requirement names capabilities, not a kernel or vendor library.

At the current production boundary, this snapshot is complete and verified. `estimate_block()` may derive cost from it without adding duration to the IR.

## Stage 4: Target binding

The target gate consumes `TargetProfile`, `DeploymentProfile`, and an evidence policy. For a GPU deployment it may select a GEMM implementation and a ring AllReduce; for an LPU it may select a different matrix primitive and collective protocol.

```text
@gemm   -> implementation gpu.cublas.gemm.bf16.v17
@reduce -> implementation nccl.all_reduce.ring.v3
abstract compute -> device gpu:0, engine compute:0
abstract network -> communicator tp0, channel 2
```

The gate checks capability, dtype/layout limits, library and ABI revisions, topology, and capacity. Missing evidence is not silently treated as zero; the resolver records interpolation, extrapolation, or analytical fallback.

## Stage 5: ConcretePlanIR

Scheduling and memory planning freeze the authoritative execution-plan envelope. The example shows a queue-centric GPU common core; a spatial or dataflow LPU also requires a typed target extension for routes, issue constraints, or other target-only semantics:

```text
buffer %partial {device=gpu:0, space=hbm, offset=0x4000, size=M*K*2}

command @c0 Launch {
  implementation = gpu.cublas.gemm.bf16.v17
  queue = gpu:0/compute:0
  writes = [%partial]
}

command @c1 Collective {
  implementation = nccl.all_reduce.ring.v3
  queue = gpu:0/network:0
  waits_for = [@c0.completed]
  reads_writes = [%partial]
}
```

Common correctness is defined by dependencies, ordering, tokens, buffers, and resource constraints. No predicted start time is required. Target-enforced cycles or slots belong to a typed extension rather than a prediction.

A complete producer's verifier must check DAG acyclicity, ordering legality, synchronization, buffer lifetime and reuse, placement, ABI compatibility, capacity, and target extensions. The current structural verifier does not cover every obligation listed here.

## Stage 6: Simulation and MachineIR

The simulation branch resolves duration intervals and applies the concrete readiness rules:

```text
@c0 predicted [0.000 ms, 0.310 ms]
@c1 predicted [0.310 ms, 0.442 ms]
critical_path = [@c0, @c1]
```

The program branch lowers the same commands to target instructions or runtime calls:

```text
machine.call @gemm_executable(...)
machine.signal @c0.completed
machine.wait @c0.completed
machine.call @nccl_all_reduce(...)
```

The time intervals are rebuildable annotations. The machine instructions and simulation events both reference `@c0` and `@c1` and their upstream lineage.

The concrete digest, time intervals, simulation events, evidence and policy fingerprints, and diagnostics form a `TimelineBundle`. The bundle references the authoritative command plan and cannot create its own schedule.

## Stage 7: Observation and recalibration

Runtime profiling records actual completion and counters:

```text
Observation(command=@c0, duration=0.337 ms, source=runtime-profiler-r8)
Observation(command=@c1, duration=0.151 ms, source=runtime-profiler-r8)
```

Normalization produces an immutable `ObservationSet`. Calibration may publish a new target-scoped evidence revision, after which the planner can re-estimate and generate a new plan. The original IR, cost view, simulation report, and program artifact remain reproducible.

## Design constraints established by the example

This illustrative example requires each stage to eliminate one class of unknown, introduce only the information it owns, and preserve stable lineage. Exact work never becomes a fitted parameter; target cost never becomes model semantics; predicted time never becomes an execution dependency; and runtime observation never mutates an old plan. These constraints become implementation evidence only after production producers, verifiers, and cross-consumer tests exist.
