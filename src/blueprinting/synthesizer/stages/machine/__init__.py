"""MachineIR stage: one target plugin's instructions, sections, entry points, and ABI."""

from .ir import (
    MachineEntryPoint,
    MachineInstruction,
    MachineIR,
    MachineOpcode,
    MachineSection,
    MachineSectionKind,
)

__all__ = [
    "MachineEntryPoint",
    "MachineInstruction",
    "MachineIR",
    "MachineOpcode",
    "MachineSection",
    "MachineSectionKind",
]
