"""Target-specific machine program contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from blueprinting.schema.codec import enum_type, record_type
from blueprinting.schema.frozen import FrozenDict

from ..errors import DiagnosticBag, VerificationReport
from ..ids import CommandId, InstructionId, Lineage
from .common import (
    CanonicalIRMixin,
    IRHeader,
    SchemaVersion,
    frozen_map,
    is_content_digest,
    make_header,
    reject_reserved_attributes,
    require_instance,
    typed_tuple,
    verify_ordered_dag,
    verify_unique_ids,
)


@record_type("compiler.machine.opcode")
@dataclass(frozen=True, order=True)
class MachineOpcode:
    dialect: str
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.dialect, str) or not isinstance(self.name, str):
            raise TypeError("machine opcode dialect and name must be strings")
        if not self.dialect or not self.name:
            raise ValueError("machine opcode dialect and name must not be empty")

    def __str__(self) -> str:
        return f"{self.dialect}.{self.name}"


@record_type("compiler.machine.instruction")
@dataclass(frozen=True)
class MachineInstruction:
    id: InstructionId
    opcode: MachineOpcode
    dependencies: tuple[InstructionId, ...]
    operands: FrozenDict
    lineage: Lineage
    source_command: CommandId | None = None
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_instance(self.id, InstructionId, "machine instruction ID")
        require_instance(self.opcode, MachineOpcode, "machine opcode")
        require_instance(self.lineage, Lineage, "machine instruction lineage")
        if self.source_command is not None:
            require_instance(self.source_command, CommandId, "machine source command")
        object.__setattr__(
            self,
            "dependencies",
            typed_tuple(self.dependencies, InstructionId, "machine instruction dependencies"),
        )
        object.__setattr__(self, "operands", frozen_map(self.operands))
        object.__setattr__(self, "attributes", frozen_map(self.attributes))


@enum_type("compiler.machine.section_kind")
class MachineSectionKind(Enum):
    CODE = "code"
    DATA = "data"
    DESCRIPTOR = "descriptor"
    METADATA = "metadata"


@record_type("compiler.machine.section")
@dataclass(frozen=True)
class MachineSection:
    name: str
    kind: MachineSectionKind
    instructions: tuple[MachineInstruction, ...] = ()
    data: bytes = b""
    alignment_bytes: int = 1
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_instance(self.kind, MachineSectionKind, "machine section kind")
        object.__setattr__(
            self,
            "instructions",
            typed_tuple(self.instructions, MachineInstruction, "machine section instructions"),
        )
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("machine section name must not be empty")
        if not isinstance(self.data, bytes):
            raise TypeError("machine section data must be bytes")
        if (
            isinstance(self.alignment_bytes, bool)
            or not isinstance(self.alignment_bytes, int)
            or self.alignment_bytes <= 0
        ):
            raise ValueError("machine section alignment must be a positive integer")
        if self.kind is MachineSectionKind.CODE and self.data:
            raise ValueError("code section cannot contain opaque data")
        if self.kind is not MachineSectionKind.CODE and self.instructions:
            raise ValueError("only code sections may contain machine instructions")


@record_type("compiler.machine.entry_point")
@dataclass(frozen=True)
class MachineEntryPoint:
    name: str
    instruction: InstructionId

    def __post_init__(self) -> None:
        require_instance(self.instruction, InstructionId, "entry-point instruction")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("machine entry-point name must not be empty")


_MACHINE_RESERVED = frozenset(
    {
        "predicted_start",
        "predicted_end",
        "predicted_duration",
        "estimated_time",
        "latency_estimate",
    }
)


@record_type("compiler.ir.machine.v1")
@dataclass(frozen=True)
class MachineIR(CanonicalIRMixin):
    """Target-owned instruction dialect before final binary/container emission."""

    SCHEMA_NAME: ClassVar[str] = "blueprinting.machine"
    SCHEMA_VERSION: ClassVar[SchemaVersion] = SchemaVersion(1, 0, 0)

    name: str
    source_concrete_digest: str
    target_fingerprint: str
    target_plugin: str
    target_abi: str
    emitter_revision: str
    program_format: str
    sections: tuple[MachineSection, ...]
    entry_points: tuple[MachineEntryPoint, ...] = ()
    attributes: FrozenDict = field(default_factory=FrozenDict)
    header: IRHeader = field(default_factory=lambda: make_header(MachineIR.SCHEMA_NAME, MachineIR.SCHEMA_VERSION))

    def __post_init__(self) -> None:
        require_instance(self.header, IRHeader, "machine header")
        identity_fields = (
            "name",
            "source_concrete_digest",
            "target_fingerprint",
            "target_plugin",
            "target_abi",
            "emitter_revision",
            "program_format",
        )
        if any(not isinstance(getattr(self, field_name), str) for field_name in identity_fields):
            raise TypeError("machine program identity fields must be strings")
        object.__setattr__(self, "sections", typed_tuple(self.sections, MachineSection, "machine sections"))
        object.__setattr__(
            self,
            "entry_points",
            typed_tuple(self.entry_points, MachineEntryPoint, "machine entry points"),
        )
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if (
            not self.header.parent_digests
            and self.header.schema_name == self.SCHEMA_NAME
            and self.header.schema_version == self.SCHEMA_VERSION
            and is_content_digest(self.source_concrete_digest)
        ):
            object.__setattr__(self, "header", self.header.with_parents(self.source_concrete_digest))

    @property
    def instructions(self) -> tuple[MachineInstruction, ...]:
        return tuple(instruction for section in self.sections for instruction in section.instructions)

    def verify(self) -> VerificationReport:
        bag = DiagnosticBag()
        self._verify_common(bag)
        identity_fields = (
            "name",
            "source_concrete_digest",
            "target_fingerprint",
            "target_plugin",
            "target_abi",
            "emitter_revision",
            "program_format",
        )
        for field_name in identity_fields:
            if not getattr(self, field_name):
                bag.error("machine.identity", f"{field_name} must not be empty", field_name)
        if not is_content_digest(self.source_concrete_digest):
            bag.error(
                "machine.source_digest", "source_concrete_digest must be a canonical digest", "source_concrete_digest"
            )
        elif self.source_concrete_digest not in self.header.parent_digests:
            bag.error("machine.parent_digest", "source digest must be retained in header", "header", "parent_digests")
        if not is_content_digest(self.target_fingerprint):
            bag.error(
                "machine.target_fingerprint",
                "target_fingerprint must be a canonical binding digest",
                "target_fingerprint",
            )
        if len({section.name for section in self.sections}) != len(self.sections):
            bag.error("machine.duplicate_section", "machine section names must be unique", "sections")

        instructions = self.instructions
        if not self.sections or not instructions:
            bag.error("machine.empty", "machine program requires a code instruction", "sections")
        verify_unique_ids(bag, instructions, lambda item: item.id, "instructions")
        verify_ordered_dag(
            bag,
            instructions,
            lambda item: item.id,
            lambda item: item.dependencies,
            "instructions",
        )
        instruction_ids = {item.id for item in instructions}
        for index, entry in enumerate(self.entry_points):
            if entry.instruction not in instruction_ids:
                bag.error(
                    "reference.unknown",
                    f"unknown entry-point instruction {entry.instruction}",
                    "entry_points",
                    str(index),
                    "instruction",
                )
        if len({item.name for item in self.entry_points}) != len(self.entry_points):
            bag.error("machine.duplicate_entry", "entry-point names must be unique", "entry_points")

        for section_index, section in enumerate(self.sections):
            path = ("sections", str(section_index))
            reject_reserved_attributes(bag, section.attributes, _MACHINE_RESERVED, *path, "attributes")
            for instruction_index, instruction in enumerate(section.instructions):
                instruction_path = path + ("instructions", str(instruction_index))
                if instruction.opcode.dialect not in {self.target_plugin, "builtin"}:
                    bag.error(
                        "machine.foreign_dialect",
                        f"opcode dialect {instruction.opcode.dialect!r} is not owned by target plugin",
                        *instruction_path,
                        "opcode",
                    )
                reject_reserved_attributes(
                    bag,
                    instruction.operands,
                    _MACHINE_RESERVED,
                    *instruction_path,
                    "operands",
                )
                reject_reserved_attributes(
                    bag,
                    instruction.attributes,
                    _MACHINE_RESERVED,
                    *instruction_path,
                    "attributes",
                )
        reject_reserved_attributes(bag, self.attributes, _MACHINE_RESERVED, "attributes")
        return bag.report()
