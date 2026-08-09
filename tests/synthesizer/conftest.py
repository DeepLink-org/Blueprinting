from __future__ import annotations

import pytest

from blueprinting.schema import FrozenDict
from blueprinting.synthesizer.ids import (
    BufferId,
    CommandId,
    DeviceId,
    InstructionId,
    Lineage,
    MemoryRegionId,
    NodeId,
    QueueId,
    TokenId,
    ValueId,
)
from blueprinting.synthesizer.ir import (
    AbstractStorageClass,
    AccessMode,
    BufferBinding,
    BufferUse,
    CollectiveKind,
    CollectiveSpec,
    CommandKind,
    ConcreteCommand,
    ConcretePlanIR,
    DevicePlacement,
    DistributedTask,
    DistributedTaskIR,
    DistributedTaskKind,
    DistributedValue,
    ImplementationRef,
    ImplementationRequirement,
    LogicalMesh,
    MachineEntryPoint,
    MachineInstruction,
    MachineIR,
    MachineOpcode,
    MachineSection,
    MachineSectionKind,
    MemoryRegion,
    MeshAxis,
    ModelIR,
    ModelOperation,
    ModelValue,
    ObjectiveDirection,
    ObjectiveKind,
    OperationName,
    PlanBuffer,
    PlanBufferRole,
    PlanObjective,
    PlanTask,
    PlanTaskKind,
    PortablePlanIR,
    QueueKind,
    QueueSpec,
    ReductionKind,
    ResourceKind,
    ResourceRequirement,
    ShardingSpec,
    TensorType,
    ValueRole,
    WorkloadFacts,
)


@pytest.fixture
def model_ir() -> ModelIR:
    input_id = ValueId.derive("fixture", "model", "input")
    weight_id = ValueId.derive("fixture", "model", "weight")
    output_id = ValueId.derive("fixture", "model", "output")
    operation_id = NodeId.derive("fixture", "model", "matmul")
    return ModelIR(
        name="fixture-model",
        values=(
            ModelValue(input_id, TensorType((4, 8), "f16"), ValueRole.INPUT, Lineage.root(), "input"),
            ModelValue(weight_id, TensorType((8, 8), "f16"), ValueRole.PARAMETER, Lineage.root(), "weight"),
            ModelValue(output_id, TensorType((4, 8), "f16"), ValueRole.OUTPUT, Lineage.root(), "output"),
        ),
        operations=(
            ModelOperation(
                operation_id,
                OperationName("core", "matmul"),
                (input_id, weight_id),
                (output_id,),
                Lineage.root(),
            ),
        ),
        inputs=(input_id,),
        outputs=(output_id,),
        attributes=FrozenDict({"phase": "prefill"}),
    )


@pytest.fixture
def distributed_ir(model_ir: ModelIR) -> DistributedTaskIR:
    source_input, source_weight, source_output = (item.id for item in model_ir.values)
    input_id = ValueId.derive("fixture", "distributed", "input")
    weight_id = ValueId.derive("fixture", "distributed", "weight")
    partial_id = ValueId.derive("fixture", "distributed", "partial")
    output_id = ValueId.derive("fixture", "distributed", "output")
    compute_id = NodeId.derive("fixture", "distributed", "matmul")
    collective_id = NodeId.derive("fixture", "distributed", "all-reduce")
    mesh = LogicalMesh("tp-mesh", (MeshAxis("tp", 2),))
    return DistributedTaskIR(
        name="fixture-distributed",
        source_model_digest=model_ir.digest,
        mesh=mesh,
        values=(
            DistributedValue(
                input_id,
                TensorType((4, 8), "f16"),
                ValueRole.INPUT,
                ShardingSpec(((), ("tp",))),
                (0, 1),
                Lineage.lowered("distribute", (source_input,)),
                source_input,
            ),
            DistributedValue(
                weight_id,
                TensorType((8, 8), "f16"),
                ValueRole.PARAMETER,
                ShardingSpec((("tp",), ())),
                (0, 1),
                Lineage.lowered("distribute", (source_weight,)),
                source_weight,
            ),
            DistributedValue(
                partial_id,
                TensorType((4, 8), "f16"),
                ValueRole.ACTIVATION,
                ShardingSpec(((), ())),
                (0, 1),
                Lineage.lowered("local-matmul", (model_ir.operations[0].id,)),
            ),
            DistributedValue(
                output_id,
                TensorType((4, 8), "f16"),
                ValueRole.OUTPUT,
                ShardingSpec.replicated(2, ("tp",)),
                (0, 1),
                Lineage.lowered("all-reduce", (source_output,)),
                source_output,
            ),
        ),
        tasks=(
            DistributedTask(
                compute_id,
                DistributedTaskKind.LOCAL_COMPUTE,
                OperationName("core", "matmul"),
                (0, 1),
                (input_id, weight_id),
                (partial_id,),
                (),
                Lineage.lowered("distribute", (model_ir.operations[0].id,)),
            ),
            DistributedTask(
                collective_id,
                DistributedTaskKind.COLLECTIVE,
                OperationName("collective", "all_reduce"),
                (0, 1),
                (partial_id,),
                (output_id,),
                (compute_id,),
                Lineage.lowered("insert-collective", (model_ir.operations[0].id,)),
                collective=CollectiveSpec(
                    CollectiveKind.ALL_REDUCE,
                    (0, 1),
                    64,
                    reduction=ReductionKind.SUM,
                ),
            ),
        ),
        inputs=(input_id, weight_id),
        outputs=(output_id,),
    )


@pytest.fixture
def portable_ir(distributed_ir: DistributedTaskIR) -> PortablePlanIR:
    input_id = BufferId.derive("fixture", "portable", "input")
    weight_id = BufferId.derive("fixture", "portable", "weight")
    partial_id = BufferId.derive("fixture", "portable", "partial")
    output_id = BufferId.derive("fixture", "portable", "output")
    compute_id = NodeId.derive("fixture", "portable", "matmul")
    collective_id = NodeId.derive("fixture", "portable", "all-reduce")
    return PortablePlanIR(
        name="fixture-portable",
        source_distributed_digest=distributed_ir.digest,
        strategy_fingerprint="a" * 40,
        planner_revision="fixture-planner-v1",
        tasks=(
            PlanTask(
                compute_id,
                PlanTaskKind.COMPUTE,
                OperationName("core", "matmul"),
                (),
                (input_id, weight_id),
                (partial_id,),
                (0, 1),
                WorkloadFacts(operations=512, read_bytes=192, write_bytes=64),
                Lineage.lowered("plan", (distributed_ir.tasks[0].id,)),
                resources=(ResourceRequirement(ResourceKind.COMPUTE, 1),),
                implementations=(ImplementationRequirement("matrix-multiply"),),
                concurrency_group="compute",
            ),
            PlanTask(
                collective_id,
                PlanTaskKind.COLLECTIVE,
                OperationName("collective", "all_reduce"),
                (compute_id,),
                (partial_id,),
                (output_id,),
                (0, 1),
                WorkloadFacts(read_bytes=64, write_bytes=64, message_bytes=64),
                Lineage.lowered("plan", (distributed_ir.tasks[1].id,)),
                resources=(ResourceRequirement(ResourceKind.NETWORK, 1),),
                implementations=(ImplementationRequirement("all-reduce"),),
                concurrency_group="network",
            ),
        ),
        buffers=(
            PlanBuffer(
                input_id,
                64,
                PlanBufferRole.INPUT,
                AbstractStorageClass.DEVICE_LOCAL,
                Lineage.lowered("plan-buffer", (distributed_ir.values[0].id,)),
                consumers=(compute_id,),
                alignment_bytes=16,
            ),
            PlanBuffer(
                weight_id,
                128,
                PlanBufferRole.CONSTANT,
                AbstractStorageClass.PERSISTENT,
                Lineage.lowered("plan-buffer", (distributed_ir.values[1].id,)),
                consumers=(compute_id,),
                alignment_bytes=16,
            ),
            PlanBuffer(
                partial_id,
                64,
                PlanBufferRole.COMMUNICATION,
                AbstractStorageClass.COMMUNICATION,
                Lineage.lowered("plan-buffer", (distributed_ir.values[2].id,)),
                producer=compute_id,
                consumers=(collective_id,),
                alignment_bytes=16,
            ),
            PlanBuffer(
                output_id,
                64,
                PlanBufferRole.OUTPUT,
                AbstractStorageClass.DEVICE_LOCAL,
                Lineage.lowered("plan-buffer", (distributed_ir.values[3].id,)),
                producer=collective_id,
                alignment_bytes=16,
            ),
        ),
        inputs=(input_id, weight_id),
        outputs=(output_id,),
        objectives=(PlanObjective(ObjectiveKind.LATENCY, ObjectiveDirection.MINIMIZE),),
    )


@pytest.fixture
def concrete_ir(portable_ir: PortablePlanIR) -> ConcretePlanIR:
    device_0 = DeviceId.derive("fixture", "concrete", "device-0")
    device_1 = DeviceId.derive("fixture", "concrete", "device-1")
    compute_queue = QueueId.derive("fixture", "concrete", "compute")
    collective_queue = QueueId.derive("fixture", "concrete", "collective")
    region_0 = MemoryRegionId.derive("fixture", "concrete", "hbm-0")
    region_1 = MemoryRegionId.derive("fixture", "concrete", "hbm-1")
    compute_command = CommandId.derive("fixture", "concrete", "matmul")
    collective_command = CommandId.derive("fixture", "concrete", "all-reduce")
    ready = TokenId.derive("fixture", "concrete", "partial-ready")
    input_buffer, weight_buffer, partial_buffer, output_buffer = (item.id for item in portable_ir.buffers)
    return ConcretePlanIR(
        name="fixture-concrete",
        source_portable_digest=portable_ir.digest,
        target_fingerprint="b" * 40,
        deployment_fingerprint="c" * 40,
        abi_revision="virtual-abi-v1",
        evidence_revision="fixture-evidence-v1",
        planner_revision="fixture-planner-v1",
        devices=(
            DevicePlacement(device_0, 0, "virtual:0"),
            DevicePlacement(device_1, 1, "virtual:1"),
        ),
        queues=(
            QueueSpec(compute_queue, device_0, QueueKind.COMPUTE, "compute:0"),
            QueueSpec(collective_queue, device_0, QueueKind.COLLECTIVE, "collective:0"),
        ),
        memory_regions=(
            MemoryRegion(region_0, device_0, "device-local", 1024, 16),
            MemoryRegion(region_1, device_1, "device-local", 1024, 16),
        ),
        buffers=(
            BufferBinding(input_buffer, region_0, 0, 64, 16, Lineage.lowered("bind", (input_buffer,))),
            BufferBinding(weight_buffer, region_0, 64, 128, 16, Lineage.lowered("bind", (weight_buffer,))),
            BufferBinding(partial_buffer, region_0, 192, 64, 16, Lineage.lowered("bind", (partial_buffer,))),
            BufferBinding(output_buffer, region_0, 256, 64, 16, Lineage.lowered("bind", (output_buffer,))),
        ),
        commands=(
            ConcreteCommand(
                compute_command,
                CommandKind.LAUNCH,
                (),
                compute_queue,
                ImplementationRef("virtual", "matmul", "1", "virtual-abi-v1"),
                (
                    BufferUse(input_buffer, AccessMode.READ),
                    BufferUse(weight_buffer, AccessMode.READ),
                    BufferUse(partial_buffer, AccessMode.WRITE),
                ),
                Lineage.lowered("bind-command", (portable_ir.tasks[0].id,)),
                signal_tokens=(ready,),
            ),
            ConcreteCommand(
                collective_command,
                CommandKind.COLLECTIVE,
                (compute_command,),
                collective_queue,
                ImplementationRef("virtual", "all-reduce", "1", "virtual-abi-v1"),
                (
                    BufferUse(partial_buffer, AccessMode.READ),
                    BufferUse(output_buffer, AccessMode.WRITE),
                ),
                Lineage.lowered("bind-command", (portable_ir.tasks[1].id,)),
                wait_tokens=(ready,),
            ),
        ),
    )


@pytest.fixture
def machine_ir(concrete_ir: ConcretePlanIR) -> MachineIR:
    launch_id = InstructionId.derive("fixture", "machine", "launch")
    collective_id = InstructionId.derive("fixture", "machine", "collective")
    return MachineIR(
        name="fixture-machine",
        source_concrete_digest=concrete_ir.digest,
        target_fingerprint=concrete_ir.target_fingerprint,
        target_plugin="virtual",
        target_abi=concrete_ir.abi_revision,
        emitter_revision="fixture-emitter-v1",
        program_format="virtual-json-v1",
        sections=(
            MachineSection(
                ".text",
                MachineSectionKind.CODE,
                (
                    MachineInstruction(
                        launch_id,
                        MachineOpcode("virtual", "launch"),
                        (),
                        FrozenDict({"implementation": "virtual:matmul:1"}),
                        Lineage.lowered("emit", (concrete_ir.commands[0].id,)),
                        concrete_ir.commands[0].id,
                    ),
                    MachineInstruction(
                        collective_id,
                        MachineOpcode("virtual", "collective"),
                        (launch_id,),
                        FrozenDict({"implementation": "virtual:all-reduce:1"}),
                        Lineage.lowered("emit", (concrete_ir.commands[1].id,)),
                        concrete_ir.commands[1].id,
                    ),
                ),
                alignment_bytes=16,
            ),
            MachineSection(
                ".metadata",
                MachineSectionKind.METADATA,
                data=b"fixture",
                alignment_bytes=8,
            ),
        ),
        entry_points=(MachineEntryPoint("main", launch_id),),
    )
