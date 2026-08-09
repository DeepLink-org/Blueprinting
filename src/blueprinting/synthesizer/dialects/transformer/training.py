"""Static work analysis for a tensor-parallel Transformer block.

This module contains algebraic operation and byte counts only.  It has no
hardware throughput, duration, overlap ratio, or per-model correction term.
Selective recomputation is represented by explicit cloned invocations so its
cost can be audited in the same way as ordinary forward work.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from blueprinting.mapping import RecomputePolicy, TensorParallelCommunication, TransformerTrainingMappingSpec
from blueprinting.schema.codec import enum_type, record_type
from blueprinting.workload import TransformerModelSpec, TransformerTrainingWorkloadSpec

from ...ir import CollectiveKind
from .common import EngineKind, PhaseWork

# Keep the legacy codec namespace as a stable serialized identity.


@enum_type("compiler.analysis.training_phase")
class TrainingPhase(Enum):
    FORWARD = "forward"
    RECOMPUTE = "recompute"
    ACTIVATION_GRADIENT = "activation_gradient"
    WEIGHT_GRADIENT = "weight_gradient"
    OPTIMIZER = "optimizer"
    RECOMMUNICATION = "recommunication"


@record_type("blueprinting.transformer.primitive-invocation.v2")
@dataclass(frozen=True)
class PrimitiveInvocation:
    """One structurally selected operation in a local block program."""

    name: str
    source_layer: str
    primitive: str
    phase: TrainingPhase
    engine: EngineKind
    work: PhaseWork
    collective: CollectiveKind | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "source_layer", "primitive"):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if not isinstance(self.phase, TrainingPhase):
            raise TypeError("phase must be TrainingPhase")
        if not isinstance(self.engine, EngineKind):
            raise TypeError("engine must be EngineKind")
        if not isinstance(self.work, PhaseWork):
            raise TypeError("work must be PhaseWork")
        if self.engine is EngineKind.COLLECTIVE:
            if self.collective is None:
                raise ValueError("collective invocations require a collective kind")
        elif self.collective is not None:
            raise ValueError("local invocations cannot carry collective metadata")


@record_type("compiler.analysis.block_memory_facts.v1")
@dataclass(frozen=True)
class BlockMemoryFacts:
    """Storage quantities for one local tensor-parallel block shard."""

    weights: int
    activation_working: int
    activation_storage: int
    activation_checkpoint: int
    weight_gradients: int
    weight_gradients_unsharded: int
    activation_gradients: int
    optimizer: int

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True)
class _Communication:
    kind: CollectiveKind
    work: PhaseWork


@dataclass(frozen=True)
class _TrainingContext:
    workload: TransformerTrainingWorkloadSpec
    mapping: TransformerTrainingMappingSpec

    @property
    def bytes_per_element(self) -> int:
        return self.workload.bytes_per_element

    @property
    def microbatch_size(self) -> int:
        return self.workload.microbatch_size

    @property
    def tensor_parallel(self) -> int:
        return self.mapping.tensor_parallel

    @property
    def data_parallel(self) -> int:
        return self.mapping.data_parallel

    @property
    def optimizer_sharding(self) -> bool:
        return self.mapping.optimizer_sharding

    @property
    def fused_activation(self) -> bool:
        return self.mapping.fused_activation

    @property
    def recompute(self) -> RecomputePolicy:
        return self.mapping.recompute

    @property
    def sequence_parallel_all_gather_redo(self) -> bool:
        return self.mapping.sequence_parallel_all_gather_redo

    @property
    def tensor_parallel_communication(self) -> TensorParallelCommunication:
        return self.mapping.tensor_parallel_communication


@dataclass(frozen=True)
class _Layer:
    name: str
    primitive: str
    engine: EngineKind
    forward: PhaseWork
    activation_gradient: PhaseWork
    weight_gradient: PhaseWork
    optimizer: PhaseWork
    recompute: bool
    recommunicate: bool
    weight_bytes: int
    activation_bytes: int
    output_bytes: int
    weight_gradient_bytes: int
    weight_gradient_unsharded_bytes: int
    activation_gradient_bytes: int
    optimizer_bytes: int
    activation_reused: bool = False
    activation_stored: bool = True
    output_stored: bool = True
    forward_communication: _Communication | None = None
    gradient_communication: _Communication | None = None
    recompute_communication: _Communication | None = None


def _work(operations: int = 0, read: int = 0, write: int = 0, message: int = 0) -> PhaseWork:
    return PhaseWork(operations=operations, read_bytes=read, write_bytes=write, message_bytes=message)


def _generic_layer(
    *,
    name: str,
    primitive: str,
    engine: EngineKind,
    bytes_per_element: int,
    forward_operations: int,
    activation_gradient_operations: int,
    weight_gradient_operations: int = 0,
    input_elements: int,
    output_elements: int,
    activation_elements: int,
    activation_gradient_elements: int,
    weight_elements: int = 0,
    weight_gradient_elements: int = 0,
    optimizer_elements: int = 0,
    optimizer_shards: int = 1,
    recompute: bool = False,
    activation_reused: bool = False,
    activation_stored: bool = True,
    output_stored: bool = True,
    forward_memory: int | None = None,
    activation_gradient_memory: int | None = None,
) -> _Layer:
    weight_bytes = weight_elements * bytes_per_element
    weight_gradient_bytes = weight_gradient_elements * bytes_per_element // optimizer_shards
    weight_gradient_unsharded_bytes = weight_gradient_elements * 4
    master_copy = weight_elements * 4 if bytes_per_element < 4 else 0
    optimizer_bytes = (master_copy + optimizer_elements * 4) // optimizer_shards

    forward_read = (input_elements + weight_elements) * bytes_per_element
    forward_write = output_elements * bytes_per_element
    if forward_memory is not None:
        forward_read, forward_write = forward_memory, 0

    agrad_read = (weight_elements + activation_gradient_elements) * bytes_per_element
    agrad_write = activation_elements * bytes_per_element
    if activation_gradient_memory is not None:
        agrad_read, agrad_write = activation_gradient_memory, 0

    if weight_elements == 0:
        # Weightless primitives do not execute a weight-gradient memory phase.
        wgrad_read = 0
        wgrad_write = 0
    else:
        wgrad_read = (activation_elements + activation_gradient_elements) * bytes_per_element
        wgrad_write = weight_gradient_elements * bytes_per_element

    optimizer_operations = weight_gradient_elements * 11 // optimizer_shards
    return _Layer(
        name=name,
        primitive=primitive,
        engine=engine,
        forward=_work(forward_operations, forward_read, forward_write),
        activation_gradient=_work(activation_gradient_operations, agrad_read, agrad_write),
        weight_gradient=_work(weight_gradient_operations, wgrad_read, wgrad_write),
        optimizer=_work(optimizer_operations, optimizer_bytes, 0),
        recompute=recompute,
        recommunicate=False,
        weight_bytes=weight_bytes,
        activation_bytes=activation_elements * bytes_per_element,
        output_bytes=output_elements * bytes_per_element,
        weight_gradient_bytes=weight_gradient_bytes,
        weight_gradient_unsharded_bytes=weight_gradient_unsharded_bytes,
        activation_gradient_bytes=activation_gradient_elements * bytes_per_element,
        optimizer_bytes=optimizer_bytes,
        activation_reused=activation_reused,
        activation_stored=activation_stored,
        output_stored=output_stored,
    )


def _linear(
    name: str,
    m: int,
    n: int,
    k: int,
    execution: _TrainingContext,
    *,
    recompute: bool,
    activation_reused: bool = False,
    activation_stored: bool = True,
) -> _Layer:
    operations = 2 * m * n * k
    return _generic_layer(
        name=name,
        primitive="linear",
        engine=EngineKind.MATRIX,
        bytes_per_element=execution.bytes_per_element,
        forward_operations=operations,
        activation_gradient_operations=operations,
        weight_gradient_operations=operations,
        input_elements=m * n,
        output_elements=m * k,
        activation_elements=m * n,
        activation_gradient_elements=m * k,
        weight_elements=n * k,
        weight_gradient_elements=n * k,
        optimizer_elements=2 * n * k,
        optimizer_shards=execution.data_parallel if execution.optimizer_sharding else 1,
        recompute=recompute,
        activation_reused=activation_reused,
        activation_stored=activation_stored,
    )


def _batch_matmul(
    name: str,
    batch: int,
    m: int,
    n: int,
    k: int,
    execution: _TrainingContext,
    *,
    recompute: bool,
    output_stored: bool = True,
) -> _Layer:
    return _generic_layer(
        name=name,
        primitive="batch_matmul",
        engine=EngineKind.MATRIX,
        bytes_per_element=execution.bytes_per_element,
        forward_operations=batch * 2 * m * n * k,
        activation_gradient_operations=batch * 4 * m * n * k,
        input_elements=batch * (m * n + n * k),
        output_elements=batch * m * k,
        activation_elements=batch * (m * n + n * k),
        activation_gradient_elements=batch * m * k,
        recompute=recompute,
        output_stored=output_stored,
    )


def _layer_norm(
    name: str,
    elements: int,
    hidden: int,
    execution: _TrainingContext,
    *,
    recompute: bool,
) -> _Layer:
    return _generic_layer(
        name=name,
        primitive="layer_norm",
        engine=EngineKind.VECTOR,
        bytes_per_element=execution.bytes_per_element,
        forward_operations=9 * elements,
        activation_gradient_operations=14 * elements,
        weight_gradient_operations=7 * elements,
        input_elements=elements,
        output_elements=elements,
        activation_elements=elements,
        activation_gradient_elements=elements,
        weight_elements=2 * hidden,
        weight_gradient_elements=2 * hidden,
        optimizer_elements=4 * hidden,
        optimizer_shards=execution.data_parallel if execution.optimizer_sharding else 1,
        recompute=recompute,
        activation_reused=True,
        activation_stored=False,
    )


def _fork(
    name: str,
    elements: int,
    users: int,
    execution: _TrainingContext,
    *,
    recompute: bool,
    activation_stored: bool = True,
) -> _Layer:
    layer = _generic_layer(
        name=name,
        primitive="fork",
        engine=EngineKind.VECTOR,
        bytes_per_element=execution.bytes_per_element,
        forward_operations=0,
        activation_gradient_operations=users * elements,
        input_elements=elements,
        output_elements=0,
        activation_elements=elements,
        activation_gradient_elements=0,
        recompute=recompute,
        activation_stored=activation_stored,
        forward_memory=0,
        activation_gradient_memory=elements * execution.bytes_per_element * (users + 1),
    )
    return layer


def _softmax(
    name: str,
    elements: int,
    execution: _TrainingContext,
    *,
    recompute: bool,
    output_stored: bool,
) -> _Layer:
    memory = 2 * elements * execution.bytes_per_element
    return _generic_layer(
        name=name,
        primitive="softmax",
        engine=EngineKind.VECTOR,
        bytes_per_element=execution.bytes_per_element,
        forward_operations=5 * elements,
        activation_gradient_operations=8 * elements,
        input_elements=elements,
        output_elements=elements,
        activation_elements=elements,
        activation_gradient_elements=elements,
        recompute=recompute,
        output_stored=output_stored,
        activation_gradient_memory=memory,
    )


def _dropout(
    name: str,
    elements: int,
    execution: _TrainingContext,
    *,
    recompute: bool,
    activation_stored: bool = True,
    output_stored: bool = True,
) -> _Layer:
    memory = 2 * elements * execution.bytes_per_element + elements
    layer = _generic_layer(
        name=name,
        primitive="dropout",
        engine=EngineKind.VECTOR,
        bytes_per_element=execution.bytes_per_element,
        forward_operations=elements,
        activation_gradient_operations=elements,
        input_elements=elements,
        output_elements=elements,
        activation_elements=0,
        activation_gradient_elements=0,
        recompute=recompute,
        activation_stored=activation_stored,
        output_stored=output_stored,
        forward_memory=memory,
        activation_gradient_memory=memory,
    )
    return replace(layer, activation_bytes=elements, activation_gradient_bytes=elements)


def _gelu(
    name: str,
    elements: int,
    execution: _TrainingContext,
    *,
    recompute: bool,
) -> _Layer:
    memory = 2 * elements * execution.bytes_per_element
    effective_elements = 0 if execution.fused_activation else elements
    layer = _generic_layer(
        name=name,
        primitive="gelu",
        engine=EngineKind.VECTOR,
        bytes_per_element=execution.bytes_per_element,
        forward_operations=8 * elements,
        activation_gradient_operations=13 * elements,
        input_elements=elements,
        output_elements=elements,
        activation_elements=effective_elements,
        activation_gradient_elements=effective_elements,
        recompute=recompute,
        activation_gradient_memory=memory,
    )
    return layer


def _residual(
    name: str,
    elements: int,
    execution: _TrainingContext,
    *,
    recompute: bool,
) -> _Layer:
    return _generic_layer(
        name=name,
        primitive="residual_add",
        engine=EngineKind.VECTOR,
        bytes_per_element=execution.bytes_per_element,
        forward_operations=elements,
        activation_gradient_operations=2 * elements,
        input_elements=2 * elements,
        output_elements=elements,
        activation_elements=2 * elements,
        activation_gradient_elements=elements,
        recompute=recompute,
        activation_reused=True,
        activation_stored=False,
    )


def _tp_communication(
    name: str,
    elements: int,
    execution: _TrainingContext,
    *,
    conjugate: bool,
    recompute: bool,
    recommunicate: bool,
    activation_stored: bool = True,
) -> _Layer:
    tp = execution.tensor_parallel
    bytes_per_element = execution.bytes_per_element
    message = elements * bytes_per_element
    reduction_operations = elements * (tp - 1) // tp if tp > 1 else 0
    memory = 2 * message if tp > 1 else 0
    split = execution.tensor_parallel_communication is TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER

    forward_comm = None
    gradient_comm = None
    recompute_comm = None
    if tp > 1:
        if split:
            forward_kind = CollectiveKind.REDUCE_SCATTER if conjugate else CollectiveKind.ALL_GATHER
            gradient_kind = CollectiveKind.ALL_GATHER if conjugate else CollectiveKind.REDUCE_SCATTER
            forward_ops = reduction_operations if conjugate else 0
            gradient_ops = 0 if conjugate else reduction_operations
            forward_comm = _Communication(forward_kind, _work(forward_ops, memory, 0, message))
            gradient_comm = _Communication(gradient_kind, _work(gradient_ops, memory, 0, message))
        elif conjugate:
            forward_comm = _Communication(CollectiveKind.ALL_REDUCE, _work(reduction_operations, memory, 0, message))
        else:
            gradient_comm = _Communication(CollectiveKind.ALL_REDUCE, _work(reduction_operations, memory, 0, message))

        if recommunicate and (split or conjugate):
            if split:
                kind = CollectiveKind.REDUCE_SCATTER if conjugate else CollectiveKind.ALL_GATHER
            else:
                kind = CollectiveKind.ALL_REDUCE
            recompute_comm = _Communication(kind, _work(message=message))

    if split:
        activation_bytes = message // tp
        activation_gradient_bytes = message // tp
    elif conjugate:
        activation_bytes = message
        activation_gradient_bytes = 0
    else:
        activation_bytes = 0
        activation_gradient_bytes = message

    return _Layer(
        name=name,
        primitive="tensor_parallel_collective",
        engine=EngineKind.COLLECTIVE,
        forward=PhaseWork(),
        activation_gradient=PhaseWork(),
        weight_gradient=PhaseWork(),
        optimizer=PhaseWork(),
        recompute=recompute,
        recommunicate=recommunicate,
        weight_bytes=0,
        activation_bytes=activation_bytes,
        output_bytes=message if tp > 1 else 0,
        weight_gradient_bytes=0,
        weight_gradient_unsharded_bytes=0,
        activation_gradient_bytes=activation_gradient_bytes,
        optimizer_bytes=0,
        activation_stored=activation_stored,
        forward_communication=forward_comm,
        gradient_communication=gradient_comm,
        recompute_communication=recompute_comm,
    )


def _build_layers(model: TransformerModelSpec, execution: _TrainingContext) -> tuple[_Layer, ...]:
    tp = execution.tensor_parallel
    if model.hidden_size % tp or model.feedforward_size % tp or model.attention_heads % tp:
        raise ValueError("hidden, feedforward, and attention heads must divide tensor parallelism")

    full_recompute = execution.recompute is RecomputePolicy.FULL
    attention_recompute = execution.recompute in {RecomputePolicy.FULL, RecomputePolicy.ATTENTION}
    attention_ag_redo = attention_recompute or execution.sequence_parallel_all_gather_redo
    mlp_ag_redo = full_recompute or execution.sequence_parallel_all_gather_redo
    batch_sequence = execution.microbatch_size * model.sequence_length
    activation = batch_sequence * model.hidden_size
    sequence_parallel = execution.tensor_parallel_communication is TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER
    local_residual = activation // tp if sequence_parallel else activation
    local_heads = model.attention_heads // tp
    projected = model.attention_heads * model.attention_head_size // tp

    layers = [
        _fork("attention.fork", local_residual, 2, execution, recompute=full_recompute),
        _layer_norm("attention.layer_norm", local_residual, model.hidden_size, execution, recompute=full_recompute),
        _tp_communication(
            "attention.input_collective",
            activation,
            execution,
            conjugate=False,
            recompute=attention_ag_redo,
            recommunicate=attention_ag_redo,
        ),
        _fork(
            "attention.qkv_fork",
            activation,
            3,
            execution,
            recompute=attention_ag_redo,
            activation_stored=not attention_ag_redo,
        ),
        _linear(
            "attention.query",
            batch_sequence,
            model.hidden_size,
            projected,
            execution,
            recompute=full_recompute,
            activation_reused=True,
            activation_stored=False,
        ),
        _linear(
            "attention.key",
            batch_sequence,
            model.hidden_size,
            projected,
            execution,
            recompute=full_recompute,
            activation_reused=True,
            activation_stored=False,
        ),
        _linear(
            "attention.value",
            batch_sequence,
            model.hidden_size,
            projected,
            execution,
            recompute=full_recompute,
            activation_reused=True,
            activation_stored=False,
        ),
        _batch_matmul(
            "attention.query_key",
            execution.microbatch_size * local_heads,
            model.sequence_length,
            model.attention_head_size,
            model.sequence_length,
            execution,
            recompute=attention_recompute,
            output_stored=not attention_recompute,
        ),
        _softmax(
            "attention.softmax",
            execution.microbatch_size * local_heads * model.sequence_length**2,
            execution,
            recompute=attention_recompute,
            output_stored=not attention_recompute,
        ),
        _dropout(
            "attention.probability_dropout",
            execution.microbatch_size * local_heads * model.sequence_length**2,
            execution,
            recompute=attention_recompute,
            activation_stored=not attention_recompute,
        ),
        _batch_matmul(
            "attention.probability_value",
            execution.microbatch_size * local_heads,
            model.sequence_length,
            model.sequence_length,
            model.attention_head_size,
            execution,
            recompute=full_recompute,
        ),
        _linear(
            "attention.output_projection",
            batch_sequence,
            projected,
            model.hidden_size,
            execution,
            recompute=full_recompute,
        ),
        _tp_communication(
            "attention.output_collective",
            activation,
            execution,
            conjugate=True,
            recompute=full_recompute,
            recommunicate=full_recompute,
            activation_stored=False,
        ),
        _dropout("attention.output_dropout", local_residual, execution, recompute=full_recompute),
        _residual("attention.residual", local_residual, execution, recompute=full_recompute),
        _fork("mlp.fork", local_residual, 2, execution, recompute=full_recompute),
        _layer_norm("mlp.layer_norm", local_residual, model.hidden_size, execution, recompute=full_recompute),
        _tp_communication(
            "mlp.input_collective",
            activation,
            execution,
            conjugate=False,
            recompute=mlp_ag_redo,
            recommunicate=mlp_ag_redo,
        ),
        _linear(
            "mlp.expand",
            batch_sequence,
            model.hidden_size,
            model.feedforward_size // tp,
            execution,
            recompute=full_recompute,
            activation_stored=not mlp_ag_redo,
        ),
        _gelu(
            "mlp.gelu",
            model.feedforward_size * batch_sequence // tp,
            execution,
            recompute=full_recompute,
        ),
        _linear(
            "mlp.contract",
            batch_sequence,
            model.feedforward_size // tp,
            model.hidden_size,
            execution,
            recompute=full_recompute,
        ),
        _tp_communication(
            "mlp.output_collective",
            activation,
            execution,
            conjugate=True,
            recompute=full_recompute,
            recommunicate=full_recompute,
            activation_stored=False,
        ),
        _dropout("mlp.output_dropout", local_residual, execution, recompute=full_recompute),
        _residual("mlp.residual", local_residual, execution, recompute=full_recompute),
    ]
    return tuple(layers)


def _local_invocation(layer: _Layer, phase: TrainingPhase, work: PhaseWork) -> PrimitiveInvocation:
    engine = EngineKind.VECTOR if phase is TrainingPhase.OPTIMIZER else layer.engine
    return PrimitiveInvocation(
        name=f"{layer.name}.{phase.value}",
        source_layer=layer.name,
        primitive=layer.primitive,
        phase=phase,
        engine=engine,
        work=work,
    )


def _communication_invocation(
    layer: _Layer,
    phase: TrainingPhase,
    communication: _Communication,
) -> PrimitiveInvocation:
    return PrimitiveInvocation(
        name=f"{layer.name}.{phase.value}.{communication.kind.value}",
        source_layer=layer.name,
        primitive=layer.primitive,
        phase=phase,
        engine=EngineKind.COLLECTIVE,
        work=communication.work,
        collective=communication.kind,
    )


def derive_transformer_block(
    model: TransformerModelSpec,
    workload: TransformerTrainingWorkloadSpec,
    mapping: TransformerTrainingMappingSpec,
) -> tuple[tuple[PrimitiveInvocation, ...], BlockMemoryFacts]:
    """Decompose a block into explicit forward/recompute/backward work."""

    mapping.validate_workload(workload)
    execution = _TrainingContext(workload, mapping)
    layers = _build_layers(model, execution)
    invocations = []

    for layer in layers:
        if layer.engine is not EngineKind.COLLECTIVE and not layer.forward.is_empty:
            invocations.append(_local_invocation(layer, TrainingPhase.FORWARD, layer.forward))
        if layer.forward_communication is not None:
            invocations.append(_communication_invocation(layer, TrainingPhase.FORWARD, layer.forward_communication))

    for layer in layers:
        if layer.recompute and layer.engine is not EngineKind.COLLECTIVE and not layer.forward.is_empty:
            invocations.append(_local_invocation(layer, TrainingPhase.RECOMPUTE, layer.forward))
        if layer.recommunicate and layer.recompute_communication is not None:
            invocations.append(
                _communication_invocation(layer, TrainingPhase.RECOMMUNICATION, layer.recompute_communication)
            )

    for layer in reversed(layers):
        if layer.engine is not EngineKind.COLLECTIVE and not layer.activation_gradient.is_empty:
            invocations.append(_local_invocation(layer, TrainingPhase.ACTIVATION_GRADIENT, layer.activation_gradient))
        if layer.gradient_communication is not None:
            invocations.append(
                _communication_invocation(
                    layer,
                    TrainingPhase.ACTIVATION_GRADIENT,
                    layer.gradient_communication,
                )
            )
        if layer.engine is not EngineKind.COLLECTIVE and not layer.weight_gradient.is_empty:
            invocations.append(_local_invocation(layer, TrainingPhase.WEIGHT_GRADIENT, layer.weight_gradient))

    for layer in layers:
        if not layer.optimizer.is_empty:
            invocations.append(_local_invocation(layer, TrainingPhase.OPTIMIZER, layer.optimizer))

    activation_storage = 0
    activation_working = 0
    for layer in layers:
        if not layer.activation_reused:
            activation_working += layer.activation_bytes
        activation_storage += layer.activation_bytes
        if not layer.output_stored:
            activation_storage -= layer.output_bytes
        if not layer.activation_stored:
            activation_storage -= layer.activation_bytes
    if execution.recompute is RecomputePolicy.FULL:
        activation_storage = 0

    activation = execution.microbatch_size * model.sequence_length * model.hidden_size
    memory = BlockMemoryFacts(
        weights=sum(layer.weight_bytes for layer in layers),
        activation_working=activation_working,
        activation_storage=activation_storage,
        activation_checkpoint=(
            activation * execution.bytes_per_element if execution.recompute is RecomputePolicy.FULL else 0
        ),
        weight_gradients=sum(layer.weight_gradient_bytes for layer in layers),
        weight_gradients_unsharded=sum(layer.weight_gradient_unsharded_bytes for layer in layers),
        activation_gradients=sum(layer.activation_gradient_bytes for layer in layers),
        optimizer=sum(layer.optimizer_bytes for layer in layers),
    )
    return tuple(invocations), memory
