from __future__ import annotations

import importlib.util

from blueprinting.synthesizer.stages.concrete_plan.ir import ConcretePlanIR
from blueprinting.synthesizer.stages.concrete_plan.passes import (
    BindReferenceQueueTargetPass,
    BindReferenceSlotTargetPass,
)
from blueprinting.synthesizer.stages.distributed.ir import DistributedTaskIR
from blueprinting.synthesizer.stages.distributed.passes import (
    DistributeTransformerInferencePass,
    DistributeTransformerTrainingPass,
)
from blueprinting.synthesizer.stages.machine.ir import MachineIR
from blueprinting.synthesizer.stages.model.ir import ModelIR
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR
from blueprinting.synthesizer.stages.portable_plan.passes import (
    PlanTransformerInferencePass,
    PlanTransformerTrainingPass,
)


def test_stage_modules_are_the_only_definition_sites() -> None:
    stage_types = (ModelIR, DistributedTaskIR, PortablePlanIR, ConcretePlanIR, MachineIR)

    assert all(".stages." in item.__module__ and item.__module__.endswith(".ir") for item in stage_types)
    assert importlib.util.find_spec("blueprinting.synthesizer.stages.distributed") is not None
    assert importlib.util.find_spec("blueprinting.synthesizer.stages.distributed_task") is None
    assert importlib.util.find_spec("blueprinting.synthesizer.ir") is None
    assert importlib.util.find_spec("blueprinting.synthesizer.lowering") is None


def test_each_implemented_boundary_is_discoverable_from_its_target_stage() -> None:
    assert DistributeTransformerTrainingPass.contract.output_type is DistributedTaskIR
    assert DistributeTransformerInferencePass.contract.output_type is DistributedTaskIR
    assert PlanTransformerTrainingPass.contract.output_type is PortablePlanIR
    assert PlanTransformerInferencePass.contract.output_type is PortablePlanIR
    assert BindReferenceQueueTargetPass.contract.output_type is ConcretePlanIR
    assert BindReferenceSlotTargetPass.contract.output_type is ConcretePlanIR
