# ADR-0002: Separate Workload and System Domain Packages

- Date: 2026-08-09
- Status: Accepted
- Scope: domain ownership, Python package paths, frontend adapters, and system-profile naming

## Context

Blueprinting derives mappings between two independent domain inputs: a workload and a candidate system. The source tree did not express that symmetry. Transformer semantics and request contracts lived under `blueprinting.synthesizer.models`, making the derivation engine appear to own its input domain. Compute, memory, and network profiles lived in `blueprinting.analysis.cost_model`, making cost analysis appear to own the system being evaluated.

This placement obscured late binding and created the wrong dependency pressure. New workload importers would have accumulated inside the synthesizer, while new chip and interconnect abstractions would have accumulated inside a cost estimator. It also made it difficult to distinguish system facts from analysis policy and a limited evidence profile from the planned `ArchitectureBlueprint`.

## Decision drivers

- Make workload and system explicit, symmetric top-level inputs to exploration.
- Keep canonical derivation mechanics separate from domain contracts.
- Prevent cost analysis from owning chip, memory, or interconnect semantics.
- Keep target binding late and prevent system facts from leaking into workload state.
- Preserve canonical tags and target-neutral derivation digests during source reorganization.
- Avoid compatibility re-exports that would leave ownership ambiguous.

## Considered alternatives

**Leave workload types in `synthesizer.models`.** This minimizes imports but makes the synthesizer own both its input and its transformation semantics.

**Leave system profiles in `analysis.cost_model`.** This preserves a small module count but couples the object being evaluated to one evaluation policy.

**Add facade packages that re-export the old implementations.** This creates attractive new paths without moving authority; both old and new packages would remain plausible sources of truth.

**Introduce the complete `ArchitectureBlueprint` immediately.** The current data and consumers do not yet support component hierarchy, NoC, power/area/cost, design variables, legality, or physical deployment. Naming the limited profile as the final architecture contract would overstate implementation maturity.

## Decision

Create two authoritative top-level domain packages:

- `blueprinting.workload` owns target-neutral model semantics, request scenarios, and logical mapping intent. It currently contains the typed Transformer training and inference contracts.
- `blueprinting.system` owns chip compute engines, memory, interconnect tiers, collective volume rules, and the aggregate `SystemProfile` imported from the retained system data.

Move workload-to-canonical-state adapters into `blueprinting.synthesizer.frontend`. These adapters construct `ModelIR` and explicit `SynthesisSession` bindings; workload contracts themselves do neither. Lowering consumes workload contracts but remains owned by the synthesizer.

Rename `HardwareProfile` to `SystemProfile` and remove its re-export from `blueprinting.analysis`. Analysis selects whether to apply profile efficiency evidence through an explicit policy argument; the system package does not import or choose `CalibrationMode`.

Do not provide `blueprinting.synthesizer.models` or `blueprinting.analysis.SystemProfile` compatibility facades. The legacy `blueprinting.types.system` package remains only for the retained calculator path and is not an admissible dependency for new formal-analysis code.

Preserve the existing `compiler.transformer.*` and `compiler.analysis.*` codec tags, including `compiler.analysis.hardware_profile.v1`. They are opaque wire identities. The workload and system record fields remain unchanged, so canonical JSON and target-neutral plan digests remain stable.

`SystemProfile` is explicitly an evidence-bearing compute/memory/network adapter. It is not the future hierarchical `ArchitectureBlueprint`, a deployment description, or a target binding.

## Consequences

- Domain ownership is visible from imports: workload, system, synthesis frontend, and analysis have distinct paths.
- Framework/model importers can grow under `workload` without becoming derivation passes.
- Chip and interconnect contracts can evolve under `system` without being tied to roofline or database providers.
- Existing Python callers must replace old package paths and the `HardwareProfile` class name.
- Existing canonical JSON remains readable because codec tags and fields are preserved; pickle/module-path compatibility is not supported.
- The current logical execution specs still combine workload scenario and mapping intent. Further separation into workload scenario and mapping strategy requires a later ADR if it changes serialized contracts.

## Migration

1. Import Transformer contracts from `blueprinting.workload`.
2. Import `build_transformer_*_model_ir()` and synthesis-session helpers from `blueprinting.synthesizer.frontend`.
3. Import `SystemProfile` and chip/interconnect component profiles from `blueprinting.system`.
4. Keep costing APIs in `blueprinting.analysis`; pass a `SystemProfile` explicitly.
5. Do not add new dependencies on `blueprinting.types.system` or recreate compatibility exports under the old paths.

## Validation

- Package-boundary tests require both top-level domain packages and reject `blueprinting.synthesizer.models`.
- Ownership tests require frontend builders to be absent from `workload` and `SystemProfile` to be absent from `analysis`.
- System tests validate chip, memory, interconnect, capacity, policy selection, and canonical round-trip behavior.
- Existing golden IR snapshots and training/inference baseline gates verify that the source move does not change target-neutral semantics or estimates.
- Ruff, the full pytest suite, bilingual documentation parity, and strict MkDocs builds remain release gates.

## Status

Accepted on 2026-08-09. This ADR governs the initial workload/system package split. The complete architecture-blueprint schema and any split of logical execution specs remain separate future decisions.
