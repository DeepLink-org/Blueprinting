"""Torch-specific event classes for blueprinting."""

from dataclasses import dataclass
from typing import Any

import torch

from .base import Event

__all__ = [
    "TorchTensorDef",
    "TorchEvent",
    "ForwardStartEvent",
    "ForwardEndEvent",
    "BackwardStartEvent",
    "BackwardEndEvent",
]


def _get_fullname(m):
    """Get full module name."""
    return f"{m.__module__}.{m.__class__.__name__}"


@dataclass
class TorchTensorDef:
    """Tensor definition for torch events.

    Note: This is separate from types.tensor.TensorDef as it's specifically
    for capturing torch tensor shapes in events.
    """

    shape: tuple = ()
    dtype: Any = None

    def __repr__(self):
        return f"TorchTensorDef({self.shape}, {self.dtype})"


class TorchEvent(Event):
    """Torch-specific event."""

    def __init__(self, name, module, inputs, params={}):
        self.name = name
        self.module = _get_fullname(module)
        try:
            if isinstance(inputs, torch.Tensor):
                self.inputs = [TorchTensorDef(tuple(inputs.shape), inputs.dtype)]
            else:
                self.inputs = [TorchTensorDef(tuple(x.shape), x.dtype) for x in inputs]
        except:
            self.inputs = None
        self.params = params

    def __repr__(self) -> str:
        return f'{self.name}("{self.module}", {self.inputs}, {self.params})'


class ForwardStartEvent(TorchEvent):
    """Forward pass start event."""

    def __init__(self, m, inputs, params={}) -> None:
        super().__init__("ForwardStartEvent", m, inputs, params=params)


class ForwardEndEvent(TorchEvent):
    """Forward pass end event."""

    def __init__(self, m, outputs, params={}) -> None:
        super().__init__("ForwardEndEvent", m, outputs, params=params)


class BackwardStartEvent(TorchEvent):
    """Backward pass start event."""

    def __init__(self, m, inputs, params={}) -> None:
        super().__init__("BackwardStartEvent", m, inputs, params=params)


class BackwardEndEvent(TorchEvent):
    """Backward pass end event."""

    def __init__(self, m, outputs, params={}) -> None:
        super().__init__("BackwardEndEvent", m, outputs, params=params)
