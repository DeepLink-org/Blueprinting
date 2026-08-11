# IR Derivation Visualization and Replay Audit

IR Explorer turns one completed, verified formal derivation into an interactive, exportable, rebuildable debugging view. It consumes canonical snapshots, pass contracts and records, and typed lineage. It never mutates a representation or writes UI layout, predicted time, or cost back into an IR.

## Unified five-stage view

The workbench exposes five stage slots in canonical order:

```text
ModelIR -> DistributedTaskIR -> PortablePlanIR -> ConcretePlanIR -> MachineIR
```

The current training production path produces the first three. `ConcretePlanIR` and `MachineIR` have no production producers, so ordinary runs show an explicit unavailable state. A valid five-stage debug bundle can be verified and opened by the same viewer. This consumer capability does not imply that target binding, scheduling, or emission exists.

Each adapter projects only structure owned by its layer: Model operations, values, and SSA dataflow; Distributed tasks, values, mesh/ranks, sharding, and collectives; Portable tasks, buffers, exact workload, resource requirements, and dependencies; Concrete commands, devices/queues, buffer allocation, and synchronization; and Machine sections, instructions, entry points, and dependencies.

Every layer also has two read-only textual forms. The short form is typed pseudo-syntax for following a derivation: it retains the schema, interface, key bindings, and layer-defining entities while compressing repeated tasks. The detailed form enumerates canonical typed fields such as stable IDs, dependencies, workload facts, resource and implementation requirements, placement, ABI, and lineage. Both are rebuilt from the immutable IR. Neither is a new wire format nor a replacement for canonical JSON and its digest.

Each layer first explains the question it answers, the current snapshot result, and the semantics it does not own. The structure view does not use a semantically arbitrary force layout; it uses a stable two-dimensional semantic matrix. Columns are derivation phases ordered from left to right, rows are fixed `subsystem × entity kind` lanes, and missing combinations remain empty. Distributed and portable Transformer views can therefore compare attention, MLP, compute, and collective work across phases on consistent horizontal lanes. Structural dependencies are subdued by default and emphasized only for the hovered adjacency. Semantic-group and search controls expose a local expansion, with at most 1000 entities available for inspection.

Adjacent boundaries default to a three-column `Source | Pass | Target` lowering table. Source and Pass span one rule instance while Target lists each expanded semantic group. One outer rounded `span` carries each complete expression and wraps with its text; continuous nested spans segment the name, parameters, and punctuation. Structure, type, topology, exact workload, and mapping parameters use distinct semantic colors, for example `[MLP.forward([ranks=2,][ops=4304896])]`. Both levels use inline wrapping and `box-decoration-break: clone`, so each visual line's highlight follows its text instead of creating a large cell-wide border. Entity type controls the name and outer base color but is not rendered as a separate head.

Short expressions do not display long hashes from content-addressed IDs directly. For unnamed `value:*`, `node:*`, and `buffer:*` entities, the derived view assigns snapshot-local aliases such as `value#1`, `node#1`, and `buffer#1` in canonical graph order. These aliases never enter canonical IR, lineage, or digests; full stable IDs remain available to search, the entity inspector, mapping audit, and debug bundles.

Source and Target contain only parameters owned by that stage: Model shape/dtype, Distributed logical ranks/sharding/collective, Portable exact operations/bytes, Concrete placement/queue/implementation, and Machine opcode/operands. The middle Pass expression renders `<pass>.<rule>(relation=..., cardinality=N → M)` with its typed signature and rewrite summary as supporting text. Rule identity and signature come from the typed pass contract; observed cardinality comes from canonical lineage evidence. Raw canonical-ID mappings and the complete pass contract live in an expandable audit section. All grouping is presentation only and creates no canonical entity.

## Adjacent-stage correspondence

Cross-stage relations are accepted by `PassManager` only after the commit-gate `TransitionVerifier` resolves target `Lineage.sources`, checks `Lineage.kind` and `Lineage.transform` against the declared pass rule, and runs that rule's executable predicate. `source_value`, `source_buffer`, and `source_command` are consistency checks rather than replacement truth. Display names, container positions, and generation order are never mapping evidence.

Every adjacent boundary likewise has short and detailed lowering forms. The short form leads with the pass input/output schemas, required bindings and analyses, produced and preserved analyses, mutation model, verification, determinism and seed policy, and rewrite rules. The detailed form adds source and target digests, the transaction commit gate, analysis invalidation, and each transform's `signature / rewrite / preserves / introduces / forbids` semantics plus lineage evidence. These rules are declared by the synthesizer's typed `PassRule` contract and travel in the debug bundle; the frontend does not infer them from names. An undeclared transform must be rendered as lineage-only evidence. The correspondence table is grouped evidence for these rules; it does not define lowering semantics.

Each adjacent boundary also provides a relation table and:

- source and target coverage;
- 1:1, 1:N, N:1, and N:M cardinality;
- generated, missing, dangling, and explicit-source mismatch counts;
- pass schemas, binding and analysis requirements, verification policy, and host-side duration.

In addition to generic checks, current rules execute named claims for bound tensor types, roles, exact local buffer sizes, producer/consumer links, operation identity, logical ranks, exact workload facts, dependency topology, buffer uses, and selected implementations. Each accepted relation carries the claim evidence that was actually executed; relation discovery alone is not counted as verification.

Declared-rule failures are blocking transaction diagnostics: analysis products, observers, and checkpoints are not published. The `TransitionReport` distinguishes `structural_only`, `canonical_conformant`, and `relation_verified`, and stores canonical-normalizer conformance separately from per-relation evidence. IR Explorer exposes both facts instead of presenting reconstruction equality as a semantic proof. Every cross-stage pass must register independent executable relation invariants and a complete normalizer. An unchanged same-stage pass may be `structural_only`; any changed transition without complete executable evidence fails before commit. There is no successful unverified transition.

## Derived overlays

Canonical topology is always the base graph. Portable task cost is an opt-in overlay tagged with provider and revision. Future `TimingProjection`, simulation traces, or observations may implement the same overlay protocol, but must be addressed by IR digest and stable entity ID. An overlay cannot add dependencies, change workload facts, or affect a canonical digest.

## Trace and debug bundle

`DerivationTrace` is an application-level derived view containing stage instances, branches, canonical snapshots, pass metadata, boundary mappings, and optional overlays. Training produces one chain; inference can record prefill and representative final-decode branches under a shared ModelIR.

`blueprinting.derivation-debug-bundle.v0` is deterministic JSON. Import re-runs canonical decoding, digest verification, structural verification, stage/schema checks, parent-digest checks, and adjacency checks. Bundles are limited to 50 MiB. This is a single-run replay format, not a canonical representation, `TimelineBundle`, or cross-run diff format.

## Current boundary

The first version does not pause or single-step `PassManager`, continue lowering from a UI-edited snapshot, or compare two runs. Breakpoints, runtime-profiler correlation, Concrete producers, Machine emitters, and simulation traces still require separate contracts and implementations.

See [ADR-0003](../project/adr/0003-derivation-debug-trace.md) for the accepted decision.
