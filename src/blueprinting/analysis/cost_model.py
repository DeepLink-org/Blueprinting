"""Evidence-backed analytical cost model for PortablePlanIR.

The model has two deliberately separate modes:

``PEAK_ONLY``
    A falsifiable baseline using advertised peak rates.

``SYSTEM_EVIDENCE``
    Uses the size-dependent efficiency curves contained in a versioned system
    profile.  These curves are hardware evidence shared by every case; no
    model name, case identifier, or reference duration participates in the
    estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..synthesizer.codec import enum_type
from ..synthesizer.ir import CollectiveKind, PortablePlanIR
from ..system import SystemProfile
from ..workload import (
    RecomputePolicy,
    TensorParallelCommunication,
    TransformerExecutionSpec,
    TransformerModelSpec,
)
from .transformer_workload import (
    BlockMemoryFacts,
    EngineKind,
    PrimitiveInvocation,
    TrainingPhase,
)

# Codec tags are stable wire identities; the legacy namespace survives the
# Python package move so existing snapshots and performance evidence still load.


@enum_type("compiler.analysis.calibration_mode")
class CalibrationMode(Enum):
    PEAK_ONLY = "peak_only"
    SYSTEM_EVIDENCE = "system_evidence"


@dataclass(frozen=True)
class TaskEstimate:
    invocation: PrimitiveInvocation
    compute_seconds: float
    memory_seconds: float
    network_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class BlockEstimate:
    mode: CalibrationMode
    tasks: tuple[TaskEstimate, ...]
    forward: float
    recompute: float
    activation_gradient: float
    weight_gradient: float
    optimizer: float
    tensor_parallel_forward: float
    tensor_parallel_backward: float
    recommunication: float


@dataclass(frozen=True)
class IterationMemory:
    weights: int
    activations: int
    activation_checkpoints: int
    weight_gradients: int
    activation_gradients: int
    optimizer: int

    @property
    def total(self) -> int:
        return (
            self.weights
            + self.activations
            + self.activation_checkpoints
            + self.weight_gradients
            + self.activation_gradients
            + self.optimizer
        )


@dataclass(frozen=True)
class IterationEstimate:
    mode: CalibrationMode
    block: BlockEstimate
    memory: IterationMemory
    forward: float
    backward: float
    optimizer: float
    recompute: float
    tensor_parallel: float
    pipeline_parallel: float
    data_parallel: float
    recommunication: float
    pipeline_bubble: float
    total: float


def _task_estimate(
    invocation: PrimitiveInvocation,
    hardware: SystemProfile,
    participants: int,
    mode: CalibrationMode,
) -> TaskEstimate:
    work = invocation.work
    processor = hardware.matrix if invocation.engine is EngineKind.MATRIX else hardware.vector
    apply_efficiency = mode is CalibrationMode.SYSTEM_EVIDENCE
    compute_seconds = (
        work.operations / processor.throughput(work.operations, apply_efficiency=apply_efficiency)
        if work.operations
        else 0.0
    )
    memory_seconds = (
        work.memory_bytes / hardware.memory.throughput(work.memory_bytes, apply_efficiency=apply_efficiency)
        if work.memory_bytes
        else 0.0
    )
    local_seconds = hardware.processing_time(compute_seconds, memory_seconds)
    network_seconds = 0.0
    if invocation.engine is EngineKind.COLLECTIVE:
        if invocation.network_tier is None or invocation.collective is None:
            raise ValueError("collective invocation is missing network facts")
        network = hardware.networks[invocation.network_tier]
        network_seconds = network.time(
            invocation.collective.value,
            work.message_bytes,
            participants,
            apply_efficiency=apply_efficiency,
        )
    return TaskEstimate(
        invocation=invocation,
        compute_seconds=compute_seconds,
        memory_seconds=memory_seconds,
        network_seconds=network_seconds,
        total_seconds=local_seconds + network_seconds,
    )


def estimate_block(
    plan: PortablePlanIR,
    hardware: SystemProfile,
    mode: CalibrationMode,
) -> BlockEstimate:
    execution = plan.attributes.get("execution_spec")
    if not isinstance(execution, TransformerExecutionSpec):
        raise TypeError("portable plan is missing TransformerExecutionSpec")
    tasks = []
    totals = dict.fromkeys(TrainingPhase, 0.0)
    tp_forward = 0.0
    tp_backward = 0.0
    recommunication = 0.0
    for task in plan.tasks:
        invocation = task.attributes.get("invocation")
        if not isinstance(invocation, PrimitiveInvocation):
            raise TypeError("plan task is missing PrimitiveInvocation")
        estimate = _task_estimate(invocation, hardware, execution.tensor_parallel, mode)
        tasks.append(estimate)
        if invocation.engine is EngineKind.COLLECTIVE:
            if invocation.phase is TrainingPhase.FORWARD:
                tp_forward += estimate.total_seconds
            elif invocation.phase is TrainingPhase.ACTIVATION_GRADIENT:
                tp_backward += estimate.total_seconds
            elif invocation.phase is TrainingPhase.RECOMMUNICATION:
                recommunication += estimate.total_seconds
            else:
                raise ValueError(f"unexpected collective phase: {invocation.phase.value}")
        else:
            totals[invocation.phase] += estimate.total_seconds
    return BlockEstimate(
        mode=mode,
        tasks=tuple(tasks),
        forward=totals[TrainingPhase.FORWARD],
        recompute=totals[TrainingPhase.RECOMPUTE],
        activation_gradient=totals[TrainingPhase.ACTIVATION_GRADIENT],
        weight_gradient=totals[TrainingPhase.WEIGHT_GRADIENT],
        optimizer=totals[TrainingPhase.OPTIMIZER],
        tensor_parallel_forward=tp_forward,
        tensor_parallel_backward=tp_backward,
        recommunication=recommunication,
    )


def _iteration_memory(
    model: TransformerModelSpec,
    execution: TransformerExecutionSpec,
    block: BlockMemoryFacts,
    blocks_per_processor: int,
) -> IterationMemory:
    memory_microbatches = min(execution.microbatch_count, execution.pipeline_parallel)
    if execution.pipeline_interleaving > 1:
        pipeline_factor = memory_microbatches * (
            1 + (execution.pipeline_parallel - 1) / (execution.pipeline_interleaving * execution.pipeline_parallel)
        )
    else:
        pipeline_factor = memory_microbatches

    if execution.recompute is RecomputePolicy.FULL:
        activation_bytes = block.activation_working
        checkpoint_bytes = round(blocks_per_processor * block.activation_checkpoint * pipeline_factor)
    else:
        activation_bytes = round(
            block.activation_working + block.activation_storage * (blocks_per_processor * pipeline_factor - 1)
        )
        checkpoint_bytes = 0

    if blocks_per_processor == 1:
        weight_gradients = block.weight_gradients_unsharded
    else:
        weight_gradients = block.weight_gradients_unsharded + block.weight_gradients * (blocks_per_processor - 1)
    return IterationMemory(
        weights=block.weights * blocks_per_processor,
        activations=activation_bytes,
        activation_checkpoints=checkpoint_bytes,
        weight_gradients=weight_gradients,
        activation_gradients=block.activation_gradients,
        optimizer=block.optimizer * blocks_per_processor,
    )


def estimate_iteration(
    plan: PortablePlanIR,
    hardware: SystemProfile,
    mode: CalibrationMode = CalibrationMode.SYSTEM_EVIDENCE,
) -> IterationEstimate:
    """Apply an explicit 1F1B/interleaved schedule to a derived block plan."""

    model = plan.attributes.get("model_spec")
    execution = plan.attributes.get("execution_spec")
    block_memory = plan.attributes.get("block_memory")
    if not isinstance(model, TransformerModelSpec):
        raise TypeError("portable plan is missing TransformerModelSpec")
    if not isinstance(execution, TransformerExecutionSpec):
        raise TypeError("portable plan is missing TransformerExecutionSpec")
    if not isinstance(block_memory, BlockMemoryFacts):
        raise TypeError("portable plan is missing BlockMemoryFacts")
    if hardware.datatype != execution.datatype:
        raise ValueError("system profile datatype does not match execution datatype")

    blocks_per_processor = math.ceil(model.block_count / execution.pipeline_parallel)
    if execution.pipeline_interleaving > blocks_per_processor:
        raise ValueError("pipeline_interleaving cannot exceed blocks per processor")
    if blocks_per_processor % execution.pipeline_interleaving:
        raise ValueError("pipeline_interleaving must divide blocks per processor")
    blocks_per_chunk = blocks_per_processor // execution.pipeline_interleaving
    chunks_per_processor = execution.pipeline_interleaving
    base_blocks_per_chunk = blocks_per_chunk - 1
    block = estimate_block(plan, hardware, mode)

    activation_elements = execution.microbatch_size * model.sequence_length * model.hidden_size
    if execution.tensor_parallel_communication is TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER:
        activation_elements //= execution.tensor_parallel
    pipeline_message = activation_elements * execution.bytes_per_element
    if execution.pipeline_parallel > 1:
        pipeline_network = hardware.networks[execution.pipeline_parallel_network]
        pipeline_point_to_point = pipeline_network.time(
            "p2p",
            pipeline_message,
            2,
            apply_efficiency=mode is CalibrationMode.SYSTEM_EVIDENCE,
        )
    else:
        pipeline_point_to_point = 0.0

    base_forward = block.forward + block.tensor_parallel_forward
    edge_forward = base_forward + pipeline_point_to_point
    base_backward = (
        block.recompute
        + block.recommunication
        + block.activation_gradient
        + block.weight_gradient
        + block.tensor_parallel_backward
    )
    edge_backward = base_backward + pipeline_point_to_point
    chunk_forward = base_blocks_per_chunk * base_forward + edge_forward
    chunk_backward = base_blocks_per_chunk * base_backward + edge_backward
    chunk_time = chunk_forward + chunk_backward

    missing_blocks = (
        execution.pipeline_parallel - model.block_count % execution.pipeline_parallel
        if model.block_count % execution.pipeline_parallel
        else 0
    )
    if base_blocks_per_chunk > 0:
        bubble_reduction = missing_blocks * (base_forward + edge_forward + base_backward + edge_backward) / 2
    else:
        bubble_reduction = missing_blocks * (edge_forward + edge_backward)
    bubble_chunks = execution.pipeline_parallel - 1
    if execution.microbatch_count % execution.pipeline_parallel:
        shortage = execution.pipeline_parallel - execution.microbatch_count % execution.pipeline_parallel
        extra_bubbles = (execution.pipeline_interleaving - 1) * shortage
    else:
        extra_bubbles = 0
    pipeline_bubble = bubble_chunks * chunk_time + extra_bubbles * chunk_time - bubble_reduction

    multiplicity = blocks_per_processor * execution.microbatch_count
    forward = multiplicity * block.forward
    backward = multiplicity * (block.activation_gradient + block.weight_gradient)
    recompute = multiplicity * block.recompute
    optimizer = blocks_per_processor * block.optimizer
    tensor_parallel = multiplicity * (block.tensor_parallel_forward + block.tensor_parallel_backward)
    recommunication = multiplicity * block.recommunication
    pipeline_parallel = (
        execution.microbatch_count * chunks_per_processor * pipeline_point_to_point * 2
        if execution.pipeline_parallel > 1
        else 0.0
    )

    data_parallel = 0.0
    if execution.data_parallel > 1:
        network = hardware.networks[execution.data_parallel_network]
        if execution.optimizer_sharding:
            per_block = network.time(
                CollectiveKind.REDUCE_SCATTER.value,
                block_memory.weights,
                execution.data_parallel,
                apply_efficiency=mode is CalibrationMode.SYSTEM_EVIDENCE,
            ) + network.time(
                CollectiveKind.ALL_GATHER.value,
                block_memory.weights,
                execution.data_parallel,
                apply_efficiency=mode is CalibrationMode.SYSTEM_EVIDENCE,
            )
        else:
            per_block = network.time(
                CollectiveKind.ALL_REDUCE.value,
                block_memory.weights,
                execution.data_parallel,
                apply_efficiency=mode is CalibrationMode.SYSTEM_EVIDENCE,
            )
        data_parallel = blocks_per_processor * per_block

    total = (
        forward
        + backward
        + optimizer
        + recompute
        + tensor_parallel
        + pipeline_parallel
        + data_parallel
        + recommunication
        + pipeline_bubble
    )
    return IterationEstimate(
        mode=mode,
        block=block,
        memory=_iteration_memory(model, execution, block_memory, blocks_per_processor),
        forward=forward,
        backward=backward,
        optimizer=optimizer,
        recompute=recompute,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        data_parallel=data_parallel,
        recommunication=recommunication,
        pipeline_bubble=pipeline_bubble,
        total=total,
    )
