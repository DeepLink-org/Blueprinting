# Vidur Raw Component-Profile Alignment

This experiment asks a deliberately falsifiable question: **after Blueprinting independently lowers an inference phase and evaluates the resulting work, how close is its component cost to a compatible Vidur profile?** Vidur is a reference result, not an implementation dependency or a latency provider for the production analysis path.

## Experimental boundary

The order is fixed:

```text
Transformer semantics + mapping + phase context
  -> Blueprinting ModelIR
  -> Blueprinting DistributedTaskIR
  -> Blueprinting PortablePlanIR
  -> Blueprinting peak-only and system-evidence costs
  -> freeze plan digests and Blueprinting estimates
  -> exact Vidur baseline lookup
  -> coverage and error report
```

`InferenceCostProvider.resolve()` is the extension point for an admissible Blueprinting performance database or hardware simulator. `InferenceBaseline.lookup()` is the external-oracle interface. `VidurProfileBaseline` implements only `lookup()`, so it cannot be supplied to `estimate_inference_phase()` by accident.

The separate `VidurProfileImporter` can explicitly convert user-supplied profile rows into a `PerformanceDatabase`. That is a different workflow and policy decision: the resulting database affects costing only when its provider is deliberately installed in a `CostResolver`. This experiment continues to use `VidurProfileBaseline` only, so its oracle isolation is unchanged.

The experiment report records `oracle_read_during_lowering = false`, `oracle_read_during_costing = false`, and `fit_against_case_outputs = false`. Per-case correction factors and Vidur durations are forbidden inputs to lowering and costing.

## Comparison contract

The adapter performs exact lookup over model and hardware identity, dtype, model dimensions, maximum sequence length, TP degree, batch/token shape, phase, context, attention backend, and cache block size. It also records a digest over the input CSV files and pinned upstream revision.

Vidur's decode `kv_cache_size` is the number of cached tokens before the current token. Blueprinting's decode `context_tokens` is the number of keys visible after the current token is appended. Therefore the exact relation is:

```text
vidur.kv_cache_size = blueprinting.context_tokens - 1
```

No nearest-neighbor match, interpolation, or silent zero fill is allowed. An absent or semantically incompatible component is `not-covered`.

## Reported metrics

Each phase report exposes:

- component count, matched count, and semantic coverage;
- Blueprinting's complete block cost;
- Blueprinting's subtotal over only matched components;
- Vidur's subtotal over the same component intersection;
- Blueprinting-estimated cost excluded from comparison;
- signed absolute and relative errors at component and comparable-subtotal levels;
- non-cancelling component MAPE and maximum component error;
- model, distributed-plan, portable-plan, hardware-evidence, and baseline revisions.

Both `PEAK_ONLY` and `SYSTEM_EVIDENCE` estimates are compared. This separates errors caused by workload lowering from errors caused by target-wide efficiency curves.

## Regression drift gate

The repository carries a minimal offline validation slice from Vidur commit `8383d2935bc62723a212090baa9f98ada206fc14`: Phi-2 on A100 with TP1, covering prefill at 128 tokens and decode at visible contexts 33 and 129. The fixture manifest records the upstream paths, Git blob IDs, MIT license, row/column projection rule, and SHA-256 of each local CSV. The full Vidur corpus remains external.

`data/validation/baseline_regression_contract.json` freezes two kinds of constraint:

- semantic-intersection and comparable-subtotal drift budgets: coverage at least 75%, comparable-subtotal MAPE at most 24%, maximum comparable-subtotal case error at most 28%, and at least 10 percentage points of improvement over peak-only;
- reviewed goldens: model/distributed/portable digests, baseline revision, comparable subtotals, per-case errors, aggregate comparable-subtotal MAPE, component MAPE, and maximum component error.

The current three-case system-evidence comparable-subtotal MAPE is 23.58%, but the non-cancelling component MAPE is 54.08% and the maximum component error is 99.50%. The lower subtotal number contains substantial cross-component cancellation and must not be presented as component accuracy or end-to-end Vidur accuracy. The gate therefore claims only provenance, semantic-coverage, and drift detection. A legitimate model improvement must update reviewed goldens while tightening causal component errors; an automatic “accept current output” update is not allowed.

Run the same mandatory CI gate locally with:

```bash
uv run pytest -m baseline_regression tests/regression
```

## Known semantic gaps

The current inference dialect models one dense-MHA, non-gated-MLP decoder template. Vidur profiles for GQA/MQA or gated-MLP models are intentionally rejected at incompatible components. The adapter does not yet consume Vidur all-reduce, send/receive, or CPU-overhead profiles.

The pinned Phi-2 rows are raw component profiles, not proof that Blueprinting reproduces Phi-2's full decoder topology. Norm placement, parallel-residual structure, fusion/layout choices, and embedding/LM-head work are not represented in the current model schema. The experiment policy therefore records `topology_equivalence = not-claimed-by-raw-component-profile-alignment`.

Vidur's public block aggregation contributes one `add_time`, while Blueprinting retains the attention residual and MLP residual as two explicit operations. Only the final MLP residual has a direct Vidur component peer; the other remains visible as excluded estimated work. This is reported as a semantic coverage gap instead of being hidden by double-counting the same baseline value.

## How alignment should improve

An error is fixed at the lowest causal layer:

1. correct operation, byte, KV, or collective derivation when the workload audit is wrong;
2. add an explicit implementation choice when FlashAttention, paged attention, fusion, or kernel family changes cost;
3. improve target-wide size/shape response models using versioned observations and held-out validation;
4. extend the normalized context for runtime, topology, or concurrency effects;
5. leave a result uncovered when the two systems do not describe the same semantic object.

The next coverage steps are gated MLP and GQA/MQA lowering, followed by collective-profile ingestion. Scheduler and queueing alignment belongs to the later serving discrete-event experiment, not this static phase baseline.
