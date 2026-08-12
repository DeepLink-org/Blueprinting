"""Hardware- and distribution-independent semantic model IR."""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import ClassVar

from blueprinting.schema.authoring import enum, record
from blueprinting.schema.frozen import FrozenDict

from ...errors import DiagnosticBag, VerificationReport
from ...ids import Lineage, NodeId, ValueId
from ...semantics import EMPTY_SEMANTIC, ModelOperationSemantic
from ..common import (
    CanonicalIRMixin,
    Effect,
    IRHeader,
    OperationName,
    SchemaVersion,
    TensorType,
    is_known_target_dialect,
    make_header,
    reject_reserved_attributes,
    verify_known_references,
    verify_ordered_dag,
    verify_unique_ids,
)


@enum("blueprinting.ir.model.value-role")
class ValueRole(Enum):
    """Semantic ownership role of a model-level SSA value."""

    INPUT = "input"
    PARAMETER = "parameter"
    CONSTANT = "constant"
    ACTIVATION = "activation"
    OUTPUT = "output"
    STATE = "state"
    KV_CACHE = "kv_cache"
    OPTIMIZER_STATE = "optimizer_state"


@record("blueprinting.ir.model.value")
class ModelValue:
    """One typed model-level SSA value with stable provenance."""

    id: ValueId
    type: TensorType
    role: ValueRole
    lineage: Lineage
    name: str = ""
    attributes: FrozenDict = field(default_factory=FrozenDict)


@record("blueprinting.ir.model.operation")
class ModelOperation:
    """One target-neutral operation with explicit dataflow and effects."""

    id: NodeId
    operation: OperationName
    inputs: tuple[ValueId, ...]
    outputs: tuple[ValueId, ...]
    lineage: Lineage
    control_dependencies: tuple[NodeId, ...] = ()
    effects: tuple[Effect, ...] = ()
    semantic: ModelOperationSemantic = EMPTY_SEMANTIC
    attributes: FrozenDict = field(default_factory=FrozenDict)


_MODEL_RESERVED = frozenset(
    {
        "model_spec",
        "tp",
        "pp",
        "dp",
        "rank",
        "device",
        "device_id",
        "queue",
        "kernel",
        "implementation_id",
        "start",
        "start_time",
        "end",
        "end_time",
        "duration",
        "latency",
        "memory_address",
        "memory_offset",
    }
)


@record("blueprinting.ir.model")
class ModelIR(CanonicalIRMixin):
    """Explicit tensor SSA graph before distribution decisions."""

    SCHEMA_NAME: ClassVar[str] = "blueprinting.model"
    SCHEMA_VERSION: ClassVar[SchemaVersion] = SchemaVersion(0, 0, 0)

    name: str
    values: tuple[ModelValue, ...]
    operations: tuple[ModelOperation, ...]
    inputs: tuple[ValueId, ...]
    outputs: tuple[ValueId, ...]
    attributes: FrozenDict = field(default_factory=FrozenDict)
    header: IRHeader = field(default_factory=lambda: make_header(ModelIR.SCHEMA_NAME, ModelIR.SCHEMA_VERSION))

    def diagnostics(self) -> VerificationReport:
        bag = DiagnosticBag()
        self._verify_common(bag)
        if not self.name:
            bag.error("model.name", "model name must not be empty", "name")

        verify_unique_ids(bag, self.values, lambda item: item.id, "values")
        verify_unique_ids(bag, self.operations, lambda item: item.id, "operations")
        value_ids = {item.id for item in self.values}
        verify_known_references(bag, self.inputs, value_ids, "inputs")
        verify_known_references(bag, self.outputs, value_ids, "outputs")
        verify_ordered_dag(
            bag,
            self.operations,
            lambda item: item.id,
            lambda item: item.control_dependencies,
            "operations",
        )

        if len(set(self.inputs)) != len(self.inputs):
            bag.error("model.duplicate_input", "model inputs must be unique", "inputs")
        if len(set(self.outputs)) != len(self.outputs):
            bag.error("model.duplicate_output", "model outputs must be unique", "outputs")

        predefined_roles = {
            ValueRole.INPUT,
            ValueRole.PARAMETER,
            ValueRole.CONSTANT,
            ValueRole.STATE,
            ValueRole.KV_CACHE,
        }
        defined = {item.id for item in self.values if item.role in predefined_roles}
        defined.update(self.inputs)
        producers = {}
        for index, operation in enumerate(self.operations):
            path = ("operations", str(index))
            if is_known_target_dialect(operation.operation):
                bag.error(
                    "model.target_dialect",
                    f"target dialect {operation.operation.dialect!r} is illegal in ModelIR",
                    *path,
                    "operation",
                )
            verify_known_references(bag, operation.inputs, value_ids, *path, "inputs")
            verify_known_references(bag, operation.outputs, value_ids, *path, "outputs")
            if len(set(operation.outputs)) != len(operation.outputs):
                bag.error("ssa.duplicate_output", "operation outputs must be unique", *path, "outputs")
            for input_id in operation.inputs:
                if input_id in value_ids and input_id not in defined:
                    bag.error("ssa.use_before_definition", f"value {input_id} is not yet defined", *path, "inputs")
            for output_id in operation.outputs:
                if output_id in producers or output_id in defined:
                    bag.error(
                        "ssa.multiple_definition", f"value {output_id} has multiple definitions", *path, "outputs"
                    )
                producers[output_id] = operation.id
                defined.add(output_id)
            reject_reserved_attributes(bag, operation.attributes, _MODEL_RESERVED, *path, "attributes")

        for index, value in enumerate(self.values):
            reject_reserved_attributes(bag, value.attributes, _MODEL_RESERVED, "values", str(index), "attributes")
            reject_reserved_attributes(
                bag,
                value.type.attributes,
                _MODEL_RESERVED,
                "values",
                str(index),
                "type",
                "attributes",
            )
            if value.role not in predefined_roles and value.id not in producers:
                bag.error("ssa.missing_definition", f"value {value.id} has no producer", "values", str(index), "id")
        for output_id in self.outputs:
            if output_id in value_ids and output_id not in defined:
                bag.error("ssa.undefined_output", f"model output {output_id} is not defined", "outputs")
        reject_reserved_attributes(bag, self.attributes, _MODEL_RESERVED, "attributes")
        return bag.report()


__all__ = ["ModelIR", "ModelOperation", "ModelValue", "ValueRole"]
