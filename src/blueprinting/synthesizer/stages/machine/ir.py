"""Target-specific machine program contract."""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import ClassVar

from blueprinting.schema.authoring import NonEmptyText, PositiveInt, enum, record
from blueprinting.schema.frozen import FrozenDict

from ...errors import DiagnosticBag, VerificationReport
from ...ids import CommandId, InstructionId, Lineage
from ..common import (
    CanonicalIRMixin,
    IRHeader,
    SchemaVersion,
    is_content_digest,
    make_header,
    reject_reserved_attributes,
    verify_ordered_dag,
    verify_unique_ids,
)


@record("blueprinting.ir.machine.opcode", order=True)
class MachineOpcode:
    """Dialect-qualified opcode owned by one target plugin."""

    dialect: NonEmptyText
    name: NonEmptyText

    def __str__(self) -> str:
        return f"{self.dialect}.{self.name}"


@record("blueprinting.ir.machine.instruction")
class MachineInstruction:
    """Target instruction with explicit dependencies, operands, and lineage."""

    id: InstructionId
    opcode: MachineOpcode
    dependencies: tuple[InstructionId, ...]
    operands: FrozenDict
    lineage: Lineage
    source_command: CommandId | None = None
    attributes: FrozenDict = field(default_factory=FrozenDict)


@enum("blueprinting.ir.machine.section-kind")
class MachineSectionKind(Enum):
    """Container role of a machine-program section."""

    CODE = "code"
    DATA = "data"
    DESCRIPTOR = "descriptor"
    METADATA = "metadata"


@record("blueprinting.ir.machine.section")
class MachineSection:
    """Aligned code or data section in a target machine program."""

    name: NonEmptyText
    kind: MachineSectionKind
    instructions: tuple[MachineInstruction, ...] = ()
    data: bytes = b""
    alignment_bytes: PositiveInt = 1
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if self.kind is MachineSectionKind.CODE and self.data:
            raise ValueError("code section cannot contain opaque data")
        if self.kind is not MachineSectionKind.CODE and self.instructions:
            raise ValueError("only code sections may contain machine instructions")


@record("blueprinting.ir.machine.entry-point")
class MachineEntryPoint:
    """Named externally addressable instruction in a machine program."""

    name: NonEmptyText
    instruction: InstructionId


_MACHINE_RESERVED = frozenset(
    {
        "predicted_start",
        "predicted_end",
        "predicted_duration",
        "estimated_time",
        "latency_estimate",
    }
)


@record("blueprinting.ir.machine")
class MachineIR(CanonicalIRMixin):
    """Target-owned instruction dialect before final binary/container emission."""

    SCHEMA_NAME: ClassVar[str] = "blueprinting.machine"
    SCHEMA_VERSION: ClassVar[SchemaVersion] = SchemaVersion(0, 0, 0)

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

    @property
    def instructions(self) -> tuple[MachineInstruction, ...]:
        return tuple(instruction for section in self.sections for instruction in section.instructions)

    def diagnostics(self) -> VerificationReport:
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


__all__ = [
    "MachineEntryPoint",
    "MachineInstruction",
    "MachineIR",
    "MachineOpcode",
    "MachineSection",
    "MachineSectionKind",
]
