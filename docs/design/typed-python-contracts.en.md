# Progressive Typed Python Contracts

Blueprinting uses a typed, algebraic subset of Python as its implementation language for formal modeling and verified derivation. This is not a second IR hierarchy and it is not an attempt to turn Python into Haskell. The design keeps ordinary Python values and tooling while making legal constructors, explicit effects, closed alternatives, and checked failure paths visible in source.

The implementation has two independent gates and one shared declaration source:

```text
Python annotations + frozen records + ADT declarations + pass contracts
             ├── runtime ContractCompiler (always available)
             └── standard mypy analysis (optional dependency)
```

Runtime declarations are authoritative. Installing mypy adds function-body, flow-sensitive, and exhaustiveness analysis; it does not change runtime semantics or introduce a second type model. Blueprinting does not require a custom mypy plugin.

## Layer 1: typed runtime values

`@record` derives frozen/slotted canonical product types and performs annotation-driven structural checks. `@adt` and `@variant` define a semantic constructor family; an explicit union alias plus `seal_adt` closes the family:

```python
@adt(wire="blueprinting.binding.workload-mode")
class WorkloadMode:
    pass


@variant("training")
class TrainingWorkload(WorkloadMode):
    pass


@variant("inference")
class InferenceWorkload(WorkloadMode):
    phase: InferencePhase


WorkloadModeVariant = TrainingWorkload | InferenceWorkload
seal_adt(WorkloadMode, WorkloadModeVariant)
```

This construction makes `InferenceWorkload` without a phase unrepresentable. The same pattern now owns portable task bodies and cost-provider support results. Derived properties such as `PlanTask.kind` are read-only projections; they are not serialized discriminators or parallel canonical schemas.

Open-world semantics remain open. Target plugins and evidence providers continue to use `Protocol` and explicit registries because their constructor set cannot be sealed by the core package.

## Layer 2: checked effects and contract compilation

Expected failure uses `Result[T, E]`; Blueprinting's common specialization is `Checked[T] = Result[T, DiagnosticSet]`. `Ok(value, warnings)` and `Err(diagnostics)` make control flow and diagnostic accumulation explicit. Exceptions remain appropriate for programmer errors and at named adapters such as `require_from_json`, `require_run`, and `CostResolver.require`.

Canonical boundaries use the checked form directly:

```python
decoded: Checked[ModelIR] = ModelIR.from_json(payload)
verified: Checked[ModelIR] = model.verify()
derived: Checked[PipelineResult[PortablePlanIR]] = manager.run(
    pipeline,
    model,
    session=session,
)
```

`ContractCompiler` imports the built-in declaration owners and compiles canonical codec entries, complete record field shapes and defaults, enum members, sealed ADT closures, and pass contracts into one immutable manifest and digest. It rejects unsealed/incomplete families, duplicate pass identities or loaded revisions, cross-stage passes without lineage rules, independent relation invariants, or a complete normalizer, and passes that can commit without output verification. A field annotation, order, constructor mode, default, enum member, or ADT constructor change therefore changes the contract digest.

Every changed successful cross-stage pass transition has executable semantic evidence. Canonical construction equality is reported separately from relation verification: `canonical_conformant` means a same-stage result matches its declared normalizer, while `relation_verified` means every materialized relation passed an independent invariant or preservation claim. A same-stage no-op may be `structural_only`; a changed transition without a law fails and cannot publish analyses, observers, or checkpoints. There is no successful `unverified` status.

Run the runtime gate with only normal project dependencies:

```bash
uv sync --locked
uv run python scripts/check_type_contracts.py
```

This gate checks loaded declarations and runtime contracts. It does not parse arbitrary Python modules or prove branch reachability inside function bodies.

## Layer 3: optional static analysis

The optional `typing` dependency group installs standard mypy:

```bash
uv sync --locked --group typing
uv run mypy src/blueprinting
```

Mypy checks annotations across modules, generic `Result` composition, union narrowing, and closed-match exhaustiveness. Strict mode is enabled first for the schema/contract foundation and selected canonical modules, then expanded progressively. The base test job deliberately runs without the `typing` group; a separate CI job runs mypy. The wheel contains `py.typed`, so downstream type checkers can consume Blueprinting's inline annotations without making mypy a runtime dependency.

No custom plugin is currently justified. A plugin would couple correctness to one checker API and duplicate runtime declaration logic. It should be reconsidered only if standard annotations cannot express a concrete, measured invariant after the runtime contract has already been defined.

## Derivation kernel and domain migrations

Transformation semantics remain pure functions of a source snapshot and explicit `SynthesisSession`. `@derivation` records the exact source/output schemas, revision, required bindings and analyses, rules, determinism policy, and canonical normalizer. The pass manager re-evaluates that normalizer and requires complete snapshot equality before commit, then runs independently declared relation invariants. Normal-form equality establishes implementation conformance; it is not counted as relation evidence.

The first migration slice removes three invalid-state patterns:

| Previous representation | Canonical algebraic representation |
|---|---|
| workload enum plus optional inference phase | `TrainingWorkload | InferenceWorkload(phase)` |
| flat TP/PP/DP strategy fields | canonical `Transformer*Parallelism` product containing typed axes and schedule ADT |
| portable task kind field | `ComputeTask | CollectiveTask | TransferTask | BarrierTask | HostTask` body |
| support status plus conditionally meaningful fields | `CostAvailable | CostUnavailable(missing_fields) | InvalidCostSupport` |
| collective kind plus optional reduction/root | `AllReduce(reduction) | ReduceScatter(reduction) | AllGather | AllToAll | Broadcast(root)` |

The existing five IR roots remain in schema epoch `0.0.0`. These changes define the unpublished initial epoch; they do not invent migration edges or compatibility history.

## Authoring rules

When adding or changing a formal type:

1. Use a frozen typed product when all fields coexist; use a sealed ADT when alternatives own different data.
2. Keep semantic tags explicit and versionless beneath the owning root schema.
3. Match closed consumers directly on an explicit union alias and close them with `typing_extensions.assert_never`; runtime does not register consumer metadata that cannot prove function-body exhaustiveness.
4. Return `Checked` for expected validation, decoding, resolution, or derivation failure; unwrap only at a named process/application boundary.
5. Keep target/provider extension points open through `Protocol` or registries rather than pretending their world is closed.
6. Run runtime contract compilation, focused positive/negative/round-trip tests, Ruff, and the optional static gate.

Metaprogramming is intentionally narrow: decorators derive mechanical structure and manifests, but do not wrap execution, inspect stack frames, synthesize hidden business rules, or replace explicit normalizers and verifiers.

Decorator surfaces are separated by audience. Plain `blueprinting.schema` and `blueprinting.synthesizer.passes` provide values/codecs and the transaction runner without exporting authoring decorators. Core schema and trusted dialect authors import `record/adt/variant` from `blueprinting.schema.authoring`; pass and target-extension authors import `derivation/relation/claim` from `blueprinting.synthesizer.passes.authoring`. Codec registries, manifest compilation, and pass registries remain implementation details.

## Current limits

This layer improves Python's ability to express compiler-like invariants, but it is not a proof assistant. Runtime closure checks confirm exact registered family membership; mypy checks closed matches it actually analyzes when they use an explicit union and `assert_never`. Neither proves a scientific formula correct. Semantic confidence still comes from independent executable invariants, canonical construction checks, negative mutation tests, lineage, evidence provenance, and reproducible experiments.

The runtime compiler currently covers built-in declarations loaded by Blueprinting. Future third-party target plugins need an explicit registration/conformance contract before they can contribute to the compiled manifest. No such general target-plugin registry is claimed as implemented today.
