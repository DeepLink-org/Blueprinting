# Calculon Comparison and Explainable Calibration

## Objective and claim boundary

This experiment independently tests three questions:

1. whether Transformer training lowering derives the correct static workload;
2. whether versioned system evidence explains the gap between a peak-only model and the reference model;
3. whether end-to-end 1F1B/interleaved schedule composition remains traceable.

Model names, Calculon durations, paper measurements, and per-case corrections are not available while compiling or estimating. Calculon and paper values enter only after those stages as comparison oracles.

Across the eight SeqSel Table 5 cases, the current system-evidence path is numerically equivalent to Calculon at floating-point precision. This means Blueprinting's workload and analytical mapping path independently reproduces the reference work, system curves, and schedule semantics; it is not evidence of zero error on real hardware or proof of broad architecture-exploration coverage. Against the paper's reported measurements, MAPE is 3.65% and maximum absolute error is 8.87%.

## Experiment identity and metric definitions

### Frozen provenance

The machine-readable report schema is `blueprinting.calculon-calibration-experiment.v0`. The current oracle is vendored Calculon `0.1.0`, whose local Python source-tree SHA-256 is `c72cf8a0a0fc9f1fb9813a2248747d6242bbc664b665abe4b5fc6b6b18f5927b`; the hardware evidence revision is `eb1eb9fcc4a6e414e85b0252c23ea9ad2730aae2`. The JSON artifact also records SHA-256 values for every model, execution, and system input, plus the ModelIR, DistributedTaskIR, PortablePlanIR, and two PassCheckpoint digests for each case.

The oracle is [Calculon](https://github.com/calculon-ai/calculon), and the paper holdout is Table 5 of Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198). Both Blueprinting estimates finish before Calculon runs; `test_oracle_runs_only_after_both_blueprinting_estimates` enforces this call order.

### Error definitions

For Blueprinting result $x_i$ and reference $r_i$, signed relative error and cross-case MAPE are

$$
e_i=100\frac{x_i-r_i}{r_i},\qquad
\operatorname{MAPE}=\frac{1}{N}\sum_{i=1}^{N}|e_i|.
$$

When both the reference and estimate for a component are zero, its error is defined as zero. A non-zero estimate against a zero reference is infinite and fails the gate. Memory additionally reports absolute byte error so a large capacity cannot hide a small byte mismatch in a percentage.

Alignment is not total-only: every case checks 19 operation/memory/message/capacity workload metrics, total memory, nine timing components, iteration total, and the paper holdout. This yields 152 workload comparisons.

### Coverage matrix

| Model | TP | PP | DP | Global batch | Microbatch | Interleave | Two modes |
|---|---:|---:|---:|---:|---:|---:|---|
| Megatron-22B | 8 | 1 | 1 | 4 | 4 | 1 | full / seqsel |
| GPT-175B | 8 | 8 | 1 | 64 | 1 | 3 | full / seqsel |
| Turing-530B | 8 | 35 | 1 | 280 | 1 | 3 | full / seqsel |
| Megatron-1T | 8 | 64 | 1 | 512 | 1 | 1 | full / seqsel |

## Experiment path

```text
model.json
   | semantic import
   v
ModelIR: transformer.decoder_training
   | transformer-distribute
   | - decompose Transformer primitives
   | - insert explicit TP collectives
   | - clone selective/full recomputation primitives
   v
DistributedTaskIR: local TP block task DAG
   | transformer-plan-work
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

### Component-level parity

| Timing component | MAPE | Maximum absolute error | Maximum absolute time error |
|---|---:|---:|---:|
| forward | 0 | 0 | 0 s |
| backward | 1.53e-14% | 2.80e-14% | 7.11e-15 s |
| optimizer | 0 | 0 | 0 s |
| recompute | 0 | 0 | 0 s |
| tensor parallel | 6.49e-15% | 1.98e-14% | 8.88e-16 s |
| pipeline parallel | 0 | 0 | 0 s |
| data parallel | 0 | 0 | 0 s |
| recommunication | 0 | 0 | 0 s |
| pipeline bubble | 2.25e-15% | 1.80e-14% | 1.78e-15 s |

All workload metrics and all eight memory totals match exactly; maximum absolute memory error is **0 bytes**. The non-zero timing differences are at the scale expected from floating-point reassociation rather than an observable model discrepancy.

The two largest seqsel cases err in the same direction. The next investigation should prioritize sequence-parallel collectives, large-scale topology, or differences in the paper's environment rather than adding model-specific coefficients.

## Reproduction and artifacts

```bash
uv run python examples/calculon_calibration.py
uv run python examples/calculon_calibration.py \
  --output examples/calculon_calibration_result.json

# Run the mandatory training and inference baseline gate used by CI.
uv run pytest -m baseline_regression tests/regression
```

The eight parametrized training regressions live in `tests/validation/test_calculon.py`. The repository-level gate additionally runs all eight cases as one experiment and evaluates `data/validation/baseline_regression_contract.json`: report schema, oracle source digest, input provenance, oracle-isolation policy, workload/component/total Calculon equivalence, memory, paper-error budgets, evidence revision, case identity, aggregate goldens, and every `PortablePlanIR` digest are frozen together. Updating a golden is a reviewed contract change; the gate has no automatic accept-current-output mode.

CI budgets cap workload, component timing, and Calculon total error at `1e-9%`, and absolute memory error at `1 byte`; current results are substantially tighter. The full machine-readable report is `examples/calculon_calibration_result.json`.

Implementation map:

- `workload/transformer.py`: typed workload and execution facts;
- `synthesizer/frontend/transformer.py`: canonical import and binding adapter;
- `synthesizer/dialects/transformer/training.py`: static operation/byte derivation;
- `synthesizer/stages/distributed/passes.py`: the ModelIR-to-DistributedTaskIR pass contract;
- `synthesizer/stages/portable_plan/passes.py`: the DistributedTaskIR-to-PortablePlanIR pass contract;
- `synthesizer/dialects/transformer/training_derivation.py`: pure derivation and pattern matching;
- `analysis/cost_model.py`: peak-only and evidence-backed views;
- `validation/calculon.py`: oracle adapter, audit, and report.
- `validation/regression.py`: strict cross-domain baseline gate and diagnostics.

This is the repository's single Blueprinting/Calculon calibration path. Future comparisons must keep oracle data unavailable until workload construction and estimation complete. See [Transformer workload derivation](../design/passes/transformer.md) for the internal transformation contracts and [performance evidence](../design/performance/index.md) for the intended provider migration.
