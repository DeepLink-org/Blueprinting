"""Events module for blueprinting.

This module provides backward compatibility. New code should use the events/ submodule directly.
"""

from .events import (
                     BackwardEndEvent,
                     BackwardStartEvent,
                     Event,
                     ForwardEndEvent,
                     ForwardStartEvent,
                     TorchEvent,
                     parse,
                     parse_tree,
)
from .events.parser import TensorDef

__all__ = [
    "Event",
    "TensorDef",
    "TorchEvent",
    "ForwardStartEvent",
    "ForwardEndEvent",
    "BackwardStartEvent",
    "BackwardEndEvent",
    "parse",
    "parse_tree",
]
