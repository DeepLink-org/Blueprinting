# Calculon Comparison and Explainable Calibration

## Objective and claim boundary

This experiment independently tests three questions:

1. whether Transformer training lowering derives the correct static workload;
2. whether versioned system evidence explains the gap between a peak-only model and the reference model;
3. whether end-to-end 1F1B/interleaved schedule composition remains traceable.

Model names, Calculon durations, paper measurements, and per-case corrections are not available while compiling or estimating. Calculon and paper values enter only after those stages as comparison oracles.

Across the eight SeqSel Table 5 cases, the current system-evidence path is numerically equivalent to Calculon at floating-point precision. This means Blueprinting's workload and analytical mapping path independently reproduces the reference work, system curves, and schedule semantics; it is not evidence of zero error on real hardware or proof of broad architecture-exploration coverage. Against the paper's reported measurements, MAPE is 3.65% and maximum absolute error is 8.87%.

## Experiment path

```text
model.json
   | semantic import
   v
ModelIR: transformer.decoder_training
   | transformer-distribute-v1
   | - decompose Transformer primitives
   | - insert explicit TP collectives
   | - clone selective/full recomputation primitives
   v
DistributedTaskIR: local TP block task DAG
   | transformer-plan-work-v1
   | - derive operations/read/write/message bytes
   | - do not bind GPU/LPU or write duration
   v
PortablePlanIR: exact WorkloadFacts
   +-- peak-only cost view
   +-- system-evidence cost view
   +-- PassCheckpoint / profiler inspection
          |
          v
1F1B schedule analysis -> comparison-only report
          |
          +-- Calculon oracle
          +-- SeqSel paper holdout
```

Each IR snapshot has an independent digest. The report stores schema and digest for both pass checkpoints so a profiler or checker can consume the exact immutable lowering boundary.

## Workload derivation rules

### Linear layers

For `Y[M,K] = X[M,N] x W[N,K]`:

```text
forward operations = 2 M N K
dgrad operations   = 2 M N K
wgrad operations   = 2 M N K
```

Activation gradient and weight gradient are separate matrix multiplications. Workload analysis does not split an opaque backward time according to ratios learned from Calculon. Memory traffic is also derived from dependencies: forward reads `X/W` and writes `Y`; dgrad reads `W/dY` and writes `dX`; wgrad reads `X/dY` and writes `dW`.

### Attention

Q/K/V projection, `Q x K^T`, softmax, dropout, `P x V`, and output projection are separate primitives. A batch matrix multiply that produces both input gradients has twice the forward gradient work:

```text
forward operations = 2 B M N K
dgrad operations   = 4 B M N K
```

Softmax, layer norm, GeLU, dropout, residual, and fork counts are declared per primitive rather than through one non-GEMM percentage.

### Recomputation

Recomputation is a graph transformation, not `forward_time x ratio`:

- `full` clones all forward compute required for restoration;
- `attn_only` clones only `QK matmul -> softmax -> probability dropout`;
- sequence-parallel all-gather redo is an independent recommunication task.

The experiment also exposes a Calculon reporting peculiarity: `block_re_flops` accumulates the running forward prefix while visiting recomputed layers, whereas `block_re_time` sums selected operation durations. The audit compares explicit recomputation operations with the latter timing semantics and does not copy the prefix counter behavior.

### Communication

TP all-reduce, reduce-scatter, and all-gather are explicit `DistributedTaskIR` collectives. Logical message bytes come from activation shapes; a network profile then applies its collective volume model:

```text
adjusted_bytes = message_bytes x volume_multiplier
adjusted_bytes += adjusted_bytes / participants x participant_offset
network_time = latency + adjusted_bytes / effective_bandwidth
```

Local reduction operations and HBM traffic required by reduce-scatter/all-reduce remain on the corresponding portable task.

### Pipeline composition

End-to-end analysis constructs the 1F1B bubble from block critical paths, blocks per stage, microbatch count, and interleaving factor. Pipeline P2P message size follows boundary-activation sharding.

The experiment rejects TP/DP overlap and offload. Those semantics require concrete command dependencies, queues, buffers, and resource-conflict analysis and must be projected from `ConcretePlanIR`, not hidden in a portable-plan overlap ratio.

## Calibration policy

Both views consume exactly the same `PortablePlanIR`:

| View | Evidence | Purpose |
|---|---|---|
| `peak_only` | peak FLOP/s, bandwidth, latency, collective structure | falsifiable naive baseline |
| `system_evidence` | the same plus operation-size, transfer-size, and network efficiency curves | target-level calibrated result |

Permitted calibration inputs are target-wide, datatype-specific response curves: matrix/vector operation-size efficiency, memory transfer-size efficiency, network bandwidth efficiency, and collective volume models.

Forbidden inputs include model or case identity, Calculon/paper total time, per-case correction factors, activation/weight-gradient ratios read from Calculon, and a fixed overlap ratio. All eight cases use one canonical hardware evidence revision.

## SeqSel Table 5 results

System profile: `a100_80g`; all eight cases share one evidence revision.

| Case | Peak-only (s) | Evidence (s) | Calculon (s) | Paper (s) | Evidence vs paper |
|---|---:|---:|---:|---:|---:|
| Megatron-22B / full | 1.1691 | 1.3956 | 1.3956 | 1.42 | -1.72% |
| Megatron-22B / seqsel | 0.9351 | 1.1367 | 1.1367 | 1.10 | +3.33% |
| GPT-175B / full | 15.4997 | 18.0288 | 18.0288 | 18.13 | -0.56% |
| GPT-175B / seqsel | 11.5500 | 13.6381 | 13.6381 | 13.75 | -0.81% |
| Turing-530B / full | 44.8563 | 49.8926 | 49.8926 | 49.05 | +1.72% |
| Turing-530B / seqsel | 30.6678 | 34.4728 | 34.4728 | 37.83 | -8.87% |
| Megatron-1T / full | 81.6511 | 90.0809 | 90.0809 | 94.42 | -4.60% |
| Megatron-1T / seqsel | 59.3347 | 66.0407 | 66.0407 | 71.49 | -7.62% |

Summary:

- peak-only versus Calculon MAPE: **12.99%**;
- system-evidence versus Calculon: equivalent within floating-point precision;
- system-evidence versus paper MAPE: **3.65%**;
- system-evidence versus paper maximum absolute error: **8.87%**.

The two largest seqsel cases err in the same direction. The next investigation should prioritize sequence-parallel collectives, large-scale topology, or differences in the paper's environment rather than adding model-specific coefficients.

## Reproduction and artifacts

```bash
uv run python examples/calculon_calibration.py
uv run python examples/calculon_calibration.py \
  --output examples/calculon_calibration_result.json

# Run the mandatory training and inference baseline gate used by CI.
uv run pytest -m baseline_regression tests/regression
```

The original eight parametrized training regressions remain in `tests/compiler/test_calculon_calibration.py`. The repository-level gate additionally runs all eight cases as one experiment and evaluates `data/validation/baseline_regression_contract.json`: workload and Calculon equivalence, memory, paper-error budgets, evidence revision, case identity, aggregate goldens, and every `PortablePlanIR` digest are frozen together. Updating a golden is a reviewed contract change; the gate has no automatic accept-current-output mode.

Implementation map:

- `compiler/models/transformer.py`: typed frontend and execution facts;
- `analysis/transformer_workload.py`: static operation/byte analysis;
- `compiler/lowering/transformer.py`: the two canonical derivation passes;
- `analysis/cost_model.py`: peak-only and evidence-backed views;
- `compiler/experiments/calculon.py`: oracle adapter, audit, and report.
- `compiler/experiments/regression.py`: strict cross-domain baseline gate and diagnostics.

This is the repository's single Blueprinting/Calculon calibration path. Future comparisons must keep oracle data unavailable until workload construction and estimation complete. See [Transformer workload derivation](../design/passes/transformer.md) for the internal transformation contracts and [performance evidence](../design/performance/index.md) for the intended provider migration.
