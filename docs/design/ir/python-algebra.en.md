# Python Algebraic IR Authoring

Blueprinting uses Python annotations, frozen/slotted dataclasses, structural pattern matching, and a small deriving layer to express canonical IR. The goal is not to imitate Haskell syntax. It is to make constructors, data relations, and invariants dominate the source while shared infrastructure derives mechanical codec, immutability, and registration behavior.

## Source layout

The five canonical representations use one consistent layout:

```text
src/blueprinting/synthesizer/stages/
├── model/
│   ├── ir.py
│   └── passes.py
├── distributed/
│   ├── ir.py
│   └── passes.py
├── portable_plan/
│   ├── ir.py
│   └── passes.py
├── concrete_plan/
│   ├── ir.py
│   └── passes.py
└── machine/
    ├── ir.py
    └── passes.py
```

`ir.py` is the only real definition site for the stage's canonical types. `passes.py` directly defines the public passes producing that stage; it may not merely forward to another `lowering` module. Transformer-specific pure derivation algebra lives in `synthesizer/dialects/transformer/*_derivation.py`.

The removed `blueprinting.synthesizer.ir` and `blueprinting.synthesizer.lowering` paths are not compatibility surfaces. Internal adapters, examples, and tests import the owning stage directly, so there is only one discoverable definition and transformation hierarchy.

## Records and ADTs

An ordinary canonical product type uses `@record`:

```python
@record("blueprinting.example.axis")
class Axis:
    name: str
    size: int
```

The decorator derives a frozen/slotted dataclass, canonical codec registration, and annotation-driven structural checks. Cross-field semantic invariants remain explicit functions or verifier rules; decorators must not hide them.

A closed sum type declares its full semantic wire namespace once on the family and retains only a short stable tag on each constructor:

```python
@adt(wire="blueprinting.ir.distributed-task.task")
class TaskBody:
    pass


@variant("local-compute")
class LocalCompute(TaskBody):
    pass


@variant("collective")
class Collective(TaskBody):
    spec: CollectiveSpecVariant
```

`collective` expands to `blueprinting.ir.distributed-task.task.collective`. The short tag remains explicit because renaming a Python class cannot implicitly change wire identity or canonical digests. Component tags do not carry independent version counters; the owning IR root controls schema compatibility. `seal_adt(TaskBody, TaskBodyVariant)` freezes the exact registered constructor set: the family root is not constructible, late variants are rejected, and runtime field checks reject unregistered subclasses. `adt_manifest(TaskBody)` returns a deterministic variant manifest for schema documentation, checkers, and tooling.

## Envelope plus sum payload

Graph-node identity, inputs, outputs, dependencies, and lineage are defined once; mutually exclusive semantics live in the ADT body:

```python
@record("blueprinting.ir.distributed-task.task-envelope")
class DistributedTask:
    id: NodeId
    body: LocalCompute | Collective | PointToPoint | Reshard | Control
    operation: OperationName
    ranks: tuple[int, ...]
    inputs: tuple[ValueId, ...]
    outputs: tuple[ValueId, ...]
    dependencies: tuple[NodeId, ...]
    lineage: Lineage
```

This removes invalid `kind + Optional payload` combinations. `CollectiveSpec` follows the same rule: `AllReduce` and `ReduceScatter` own a required reduction, `Broadcast` owns a required root, while `AllGather` and `AllToAll` own neither. Interpreters, verifiers, and derived views match on explicit union aliases. The deriving layer enforces exact runtime closure; optional mypy analysis checks match exhaustiveness when the subject is an explicit closed union and `assert_never` closes the function.

## Derivable versus explicit information

Derivable information includes immutability, slots, constructors, equality/hash, codec registration, ADT membership, basic field types, match arguments, and variant manifests.

Wire-family identity, constructor local tags, schema migrations, lineage, pass/rule identity, cross-field invariants, preservation laws, target legality, and evidence validity remain explicit.

All five roots currently belong to the initial `0.0.0` schema epoch. The default migration registry is intentionally empty: no unpublished intermediate representation is treated as compatibility history. The migration mechanism remains available for the first graduated schema boundary.

## Passes, relations, and canonical construction

The pass generic already declares input/output types. An ordinary cross-stage derivation declares its entity relations, an independent invariant for each relation, and one pure `normalize(source, session) -> target` function:

```python
def verify_decompose(
    source: ModelOperation,
    target: DistributedTask,
    context: RelationCheckContext,
) -> None:
    if source.id not in target.lineage.sources:
        raise ValueError("task does not retain its semantic source")
    if target.ranks != tuple(range(context.target_ir.mesh.size)):
        raise ValueError("task does not cover the derived logical mesh")


decompose = relation(
    "transformer-decompose",
    "Expand one semantic operation into distributed tasks",
    source=ModelOperation,
    target=DistributedTask,
    verifier=verify_decompose,
    introduces=("logical ranks", "collective tasks"),
)


def normalize(source: ModelIR, session: SynthesisSession) -> DistributedTaskIR:
    return DistributedTaskIR(...)


@derivation(
    "transformer-distribute",
    revision="1",
    bindings=(BindingAxis.WORKLOAD, BindingAxis.STRATEGY),
    rules=(decompose,),
    normalizer=normalize,
)
class DistributeTransformerTrainingPass(
    DerivationPass[ModelIR, DistributedTaskIR]
):
    ...
```

The commit gate resolves the complete lineage graph, evaluates the normalizer again, and requires exact equality with the candidate snapshot. This is canonical implementation conformance: it detects omissions and deviations across the whole schema shape. It is recorded separately from relation evidence. Each relation verifier then checks an independently stated semantic invariant such as workload conservation, legal role mapping, dependency correspondence, or target ABI compatibility. A normalizer cannot serve as its own semantic proof. `relation(..., preserves=(claim(...),))` declares both relations and named sub-obligations as ordinary immutable values; there is no parallel `@rule` decorator syntax.

`@derivation` derives `ModelIR -> DistributedTaskIR` and exact schema versions from the generic base. The decorators only build static contracts: they do not wrap `run()`, inspect the call stack, or change execution semantics.

Passes use `match` for closed ADTs and typed strategies. Registries or `singledispatch` are reserved for open target-plugin and obligation interpreters; canonical constructors are not extended through runtime monkey patching.
