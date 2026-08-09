# Inference Planning and Serving Simulation

Blueprinting now has a runnable decoder-inference slice, but its boundary is intentionally narrower than a serving-system simulator. The implemented path answers a hardware-planning question: **what work, communication, state capacity, and static latency does one prefill or decode phase point induce under a declared mapping?** It does not yet claim to predict queueing, continuous batching, or SLO tails.

![Inference planning path and serving-simulation boundary](../assets/architecture/inference-planning-path.svg)

## What we adopt from related work

[LLMCompass](https://arxiv.org/abs/2312.03134) demonstrates why LLM inference hardware evaluation needs separate software, hardware, mapping, and cost concerns, plus an explicit mapping search rather than a single closed-form model. Blueprinting adopts that separation. Its canonical representations preserve workload and mapping facts before a hardware profile or measured latency is consulted. LLMCompass's area/cost and architecture design-space machinery remains future provider and exploration work; its artifact code is not copied into the canonical IR.

[Vidur](https://github.com/microsoft/vidur) demonstrates a complementary boundary: request arrivals, replica scheduling, batching, and event progression are a discrete-event layer, while execution time is supplied by component predictors trained from profiling data. Blueprinting adopts that split. Phase plans are the stable cost subjects; a future serving simulator will schedule requests and batches against them rather than redefining Transformer work inside scheduler code.

The resulting boundary is deliberate:

| Concern | Current owner | Status |
|---|---|---|
| Transformer operation/byte/collective derivation | canonical inference analysis | Implemented slice |
| Prefill and decode specialization | workload binding + lowering passes | Implemented slice |
| KV-cache state and capacity | ModelIR effect + portable state buffer + memory view | Implemented slice |
| Analytical component cost | `HardwareProfile` fallback | Implemented slice |
| Vidur profiling CSV reuse | post-hoc exact-match baseline | Implemented experiment |
| Static decoder-block phase composition | inference application service | Implemented slice |
| Arrivals, queues, continuous batching, scheduling | serving discrete-event simulator | Planned |
| Hardware area, power, cost, and mapping search | architecture providers and exploration session | Planned |

## Implemented derivation path

The model frontend emits one phase-neutral `transformer.decoder_inference` operation with an explicit KV-cache state effect. A phase workload binding then specializes it:

```text
TransformerModelSpec
  + TransformerInferenceExecutionSpec(TP, PP, replicas, dtype, network tiers)
  + WorkloadBinding(INFERENCE, PREFILL | DECODE, batch, context)
  -> ModelIR
  -> DistributeTransformerInferencePass
  -> DistributedTaskIR
  -> PlanTransformerInferencePass
  -> PortablePlanIR
  -> estimate_inference_phase(Blueprinting cost provider | analytical model)
  -> optional post-hoc Vidur comparison
```

The component tasks are input norm, QKV projection, RoPE, KV save, attention core, output projection, TP all-reduce, residual, post-attention norm, MLP up/activation/down, a second all-reduce, and the final residual. This boundary is fine enough to inspect work conservation and broad enough to match observable kernel families in profiling systems.

Prefill binds `query_tokens = context_tokens = prompt_tokens`. Decode binds `query_tokens = 1` and treats `context_tokens` as the number of keys visible after the current token is appended. Every phase plan carries exact operations, read/write bytes, collective volume, phase, primitive, source layer, query length, context length, block weight capacity, KV capacity, and a conservative workspace buffer. It carries no duration. Costing reconstructs its task view from `PlanTask.workload` and explicit buffers; a hidden lowering object is not allowed to become a second workload truth.

## Request composition semantics

For a cohort with prompt length `S` and requested output length `O`, the current decoder-block dialect composes one prefill phase and `O-1` decode phases:

```text
prefill model time = cost(prefill(batch=B, query=S, context=S))
decode contexts = S+1, S+2, ..., S+O-1
model execution time = prefill model time + sum(cost(decode(batch=B, query=1, context=c)))
mean decode-step model time = decode total / (O-1), when O > 1
```

Compiling each decode context separately is intentional in this first slice: size-dependent efficiency need not be linear in context. The report retains every decode plan digest and latency, while the full IR snapshots are exposed for prefill and the final decode context. Because embedding, LM head, sampling, host work, and queueing are absent, these values are deliberately not named TTFT, TPOT, or E2E. This is static model-phase composition, not a serving trace simulation.

Per-device capacity is derived from the mapping. Dense MHA KV storage for one local block shard is:

```text
2 × batch × context × (hidden / TP) × bytes_per_element
```

It is multiplied by the number of blocks in one pipeline stage. Weight storage is likewise sharded by TP and partitioned by PP; pipeline-boundary buffers remain explicit. Because `PortablePlanIR` has not selected FlashAttention, paged attention, or an unfused implementation yet, working memory is reported as a conservative unfused score-materialization upper bound. Target binding must replace that bound with implementation-specific workspace. The current path requires TP to divide hidden, FFN, and head dimensions, and PP to divide the block count.

## Using Vidur as a baseline, not as the estimator

`VidurProfileBaseline.from_csv(...)` consumes user-supplied Vidur `attention.csv` and compute/MLP CSV files. The caller must pin an upstream revision, hardware identity, attention backend, and cache block size. The adapter hashes the inputs and identity into a baseline revision, converts Vidur's millisecond medians to seconds, and only returns a reference when model dimensions, maximum sequence length, TP, batch/token shape, phase, backend, block size, and context match exactly. Vidur records decode `kv_cache_size` before the current token is appended; Blueprinting records the visible context after append, so the adapter makes the explicit relation `vidur_kv_cache_size = context_tokens - 1`.

```python
from blueprinting.analysis import HardwareProfile, VidurProfileBaseline
from blueprinting.compiler.bindings import InferencePhase
from blueprinting.compiler.experiments import VidurExperimentCase, run_vidur_experiment
from blueprinting.compiler.models import TransformerInferenceExecutionSpec, TransformerModelSpec

baseline = VidurProfileBaseline.from_csv(
    attention_csv="/profiles/attention.csv",
    compute_csv="/profiles/mlp.csv",
    model_name="<exact Vidur model identity>",
    hardware_name="a100_80g",
    attention_backend="AttentionBackend.FLASH_ATTENTION",
    block_size=16,
    source_revision="<pinned Vidur commit>",
)
case = VidurExperimentCase(
    name="decode/context-128",
    model=TransformerModelSpec(...),
    execution=TransformerInferenceExecutionSpec(...),
    hardware=HardwareProfile(...),
    phase=InferencePhase.DECODE,
    batch_size=1,
    context_tokens=128,
)
report = run_vidur_experiment((case,), baseline)
```

The API boundary is intentional: an admissible internal `InferenceCostProvider` exposes `resolve()`, while an external `InferenceBaseline` exposes `lookup()`. `run_vidur_experiment()` completes lowering and both Blueprinting cost modes before calling `lookup()`. Vidur therefore cannot alter operations, bytes, dependencies, the plan digest, or the compiled latency.

Comparison is over an explicit semantic intersection. The report contains matched component count, coverage, Blueprinting's comparable subtotal, Vidur's comparable subtotal, excluded Blueprinting work, signed comparable-subtotal error, and non-cancelling component MAPE/max error. Missing records remain `not-covered`; they are never converted to zero. This matters because Vidur's public block aggregation has one `add_time`, whereas Blueprinting deliberately keeps both residual additions explicit, and the current CSV adapter does not yet ingest collective profiles.

There is no nearest-neighbor or hidden interpolation. An MHA workload requires equal query/KV head counts in both compute and attention records. Matching raw component-profile keys does not establish full decoder-topology equivalence: the current model spec does not yet encode norm placement, residual topology, or gated-MLP choice. The current production inference estimate remains entirely Blueprinting-owned; Vidur is an oracle for measuring where that estimate must improve.

Blueprinting does not vendor the full upstream profiling corpus. A minimal MIT-licensed Phi-2/A100 validation slice is retained for offline CI, with a pinned upstream commit, source blob IDs, an explicit projection rule, and local file digests. Larger experiments keep Vidur data external. The implemented `VidurProfileImporter` now normalizes units, creates raw-record IDs, and preserves the source revision/file digest when explicitly promoting profiles into a performance database. Full environment manifests and runtime/kernel identity remain required future evidence work where the upstream schema does not supply them.

## What the serving layer must add

A Vidur-like serving simulator should be a consumer of phase plans and evidence, not another Transformer estimator. Its minimum state is:

- immutable request records: arrival, prompt/output lengths, priority, and SLO;
- replica and KV allocation state;
- scheduler policy with explicit batching/chunked-prefill decisions;
- an event queue for admission, batch start/end, transfer, preemption, and completion;
- CPU/scheduler overhead evidence separated from device component evidence;
- TTFT, inter-token latency, E2E, throughput, utilization, and tail distributions.

The scheduler produces a concrete batch context and asks the cost resolver for that context. This allows vLLM-, Orca-, Sarathi-, or future LPU-oriented policies to share the same canonical workload semantics and hardware evidence interfaces.

## Current limitations

The implemented dialect covers one dense-MHA, non-gated-MLP decoder template. Embedding, LM head, sampler, explicit norm/residual topology, GQA/MQA, gated MLP, MoE, prefix caching, paged allocation, chunked prefill, speculative decoding, disaggregated prefill/decode, scheduler overhead, and resource contention are not modeled yet. PP and replica structure are not fully materialized in `DistributedTaskIR`; PP latency/memory composition is currently analytical after the local-TP block plan. Each decode context is recompiled rather than algebraically specialized from a parametric plan. `replicas` currently participates only in mapping validation and world-size accounting; reported latency and static model token rate remain single-replica views, not multi-replica serving capacity. Consequently, the current output is suitable for inspecting derivation, analytical memory fit, and first-order hardware sensitivity—not for claiming production serving SLO accuracy.
