from __future__ import annotations

from dataclasses import fields, replace

import pytest

from blueprinting.schema import FrozenDict
from blueprinting.synthesizer import BufferId, NodeId, ValueId
from blueprinting.synthesizer.stages.common import OperationName
from blueprinting.synthesizer.stages.concrete_plan.ir import ConcretePlanIR
from blueprinting.synthesizer.stages.distributed.ir import (
    AllGather,
    AllReduce,
    Broadcast,
    CollectiveKind,
    CollectiveSpec,
    DistributedTaskIR,
    ReductionKind,
    collective_kind,
    make_collective_spec,
)
from blueprinting.synthesizer.stages.machine.ir import MachineIR, MachineOpcode
from blueprinting.synthesizer.stages.model.ir import ModelIR
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR


def _codes(ir: object) -> set[str]:
    return {item.code for item in ir.diagnostics().diagnostics}  # type: ignore[attr-defined]


def test_fixture_snapshots_are_valid(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
    portable_ir: PortablePlanIR,
    concrete_ir: ConcretePlanIR,
    machine_ir: MachineIR,
) -> None:
    for ir in (model_ir, distributed_ir, portable_ir, concrete_ir, machine_ir):
        assert ir.verify().is_ok, tuple(item.render() for item in ir.diagnostics().diagnostics)


def test_model_rejects_target_dialect_and_unknown_output(model_ir: ModelIR) -> None:
    target_operation = replace(model_ir.operations[0], operation=OperationName("cuda", "matmul"))
    invalid = replace(
        model_ir,
        operations=(target_operation,),
        outputs=(ValueId.derive("fixture", "unknown-output"),),
    )

    assert {"model.target_dialect", "reference.unknown"}.issubset(_codes(invalid))


def test_model_rejects_typed_semantics_smuggled_through_attributes(model_ir: ModelIR) -> None:
    operation = replace(model_ir.operations[0], attributes=FrozenDict({"model_spec": "legacy"}))

    assert "attribute.reserved" in _codes(replace(model_ir, operations=(operation,)))


def test_distributed_rejects_physical_rank(distributed_ir: DistributedTaskIR) -> None:
    invalid_task = replace(distributed_ir.tasks[0], ranks=(0, 99))
    invalid = replace(distributed_ir, tasks=(invalid_task,) + distributed_ir.tasks[1:])

    assert "rank.unknown" in _codes(invalid)


def test_collective_adt_makes_kind_specific_fields_structural() -> None:
    all_reduce = make_collective_spec(
        CollectiveKind.ALL_REDUCE,
        (0, 1),
        64,
        reduction=ReductionKind.SUM,
    )
    broadcast = make_collective_spec(CollectiveKind.BROADCAST, (0, 1), 64, root=0)

    assert isinstance(all_reduce, AllReduce)
    assert isinstance(broadcast, Broadcast)
    assert collective_kind(all_reduce) is CollectiveKind.ALL_REDUCE
    assert {field.name for field in fields(AllGather)} == {"participants", "message_bytes"}
    assert {field.name for field in fields(AllReduce)} == {"participants", "message_bytes", "reduction"}
    assert {field.name for field in fields(Broadcast)} == {"participants", "message_bytes", "root"}
    with pytest.raises(TypeError, match="is abstract"):
        CollectiveSpec()
    with pytest.raises(ValueError, match="requires reduction"):
        make_collective_spec(CollectiveKind.ALL_REDUCE, (0, 1), 64)
    with pytest.raises(ValueError, match="does not accept reduction or root"):
        make_collective_spec(
            CollectiveKind.ALL_GATHER,
            (0, 1),
            64,
            reduction=ReductionKind.SUM,
        )
    with pytest.raises(ValueError, match="root must be one of its participants"):
        Broadcast((0, 1), 64, 2)


def test_portable_rejects_timing_smuggled_through_attributes(portable_ir: PortablePlanIR) -> None:
    invalid = replace(portable_ir, attributes=FrozenDict({"duration": 10.0}))

    assert "attribute.reserved" in _codes(invalid)


def test_portable_rejects_dialect_semantics_smuggled_through_attributes(portable_ir: PortablePlanIR) -> None:
    task = replace(portable_ir.tasks[0], attributes=FrozenDict({"source_layer": "legacy"}))

    assert "attribute.reserved" in _codes(replace(portable_ir, tasks=(task,) + portable_ir.tasks[1:]))


def test_portable_unknown_producer_is_diagnostic_not_verifier_crash(portable_ir: PortablePlanIR) -> None:
    unknown = NodeId.derive("fixture", "unknown-producer")
    invalid_buffer = replace(portable_ir.buffers[2], producer=unknown)
    invalid = replace(portable_ir, buffers=portable_ir.buffers[:2] + (invalid_buffer,) + portable_ir.buffers[3:])

    assert "reference.unknown" in _codes(invalid)


def test_concrete_plan_has_no_authoritative_timing_fields() -> None:
    field_names = {item.name for item in fields(ConcretePlanIR)}

    assert field_names.isdisjoint({"start", "start_time", "end", "end_time", "duration", "latency"})


def test_concrete_rejects_non_topological_commands(concrete_ir: ConcretePlanIR) -> None:
    invalid = replace(concrete_ir, commands=tuple(reversed(concrete_ir.commands)))

    assert "dag.not_topological" in _codes(invalid)


def test_concrete_rejects_out_of_bounds_buffer(concrete_ir: ConcretePlanIR) -> None:
    oversized = replace(concrete_ir.buffers[0], offset_bytes=1008, size_bytes=64)
    invalid = replace(concrete_ir, buffers=(oversized,) + concrete_ir.buffers[1:])

    assert "buffer.out_of_bounds" in _codes(invalid)


def test_concrete_rejects_timing_annotations(concrete_ir: ConcretePlanIR) -> None:
    invalid_command = replace(concrete_ir.commands[0], attributes=FrozenDict({"predicted_start": 1.0}))
    invalid = replace(concrete_ir, commands=(invalid_command,) + concrete_ir.commands[1:])

    assert "attribute.reserved" in _codes(invalid)


def test_machine_rejects_foreign_target_dialect(machine_ir: MachineIR) -> None:
    instruction = replace(machine_ir.instructions[0], opcode=MachineOpcode("cuda", "launch"))
    code = replace(machine_ir.sections[0], instructions=(instruction,) + machine_ir.instructions[1:])
    invalid = replace(machine_ir, sections=(code,) + machine_ir.sections[1:])

    assert "machine.foreign_dialect" in _codes(invalid)


def test_unknown_buffer_reference_is_reported(concrete_ir: ConcretePlanIR) -> None:
    use = replace(concrete_ir.commands[0].buffers[0], buffer=BufferId.derive("fixture", "unknown-buffer"))
    command = replace(concrete_ir.commands[0], buffers=(use,) + concrete_ir.commands[0].buffers[1:])
    invalid = replace(concrete_ir, commands=(command,) + concrete_ir.commands[1:])

    assert "reference.unknown" in _codes(invalid)
