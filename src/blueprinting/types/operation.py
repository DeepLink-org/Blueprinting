"""Operation and Calculation definitions for blueprinting."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import hyperparameter as hp

from .tensor import TensorDef

__all__ = [
    "Operation",
    "Calculation",
]


@dataclass
class Operation:
    """Base operation class."""

    inputs: Tuple[TensorDef, ...] = ()
    output: TensorDef = field(default_factory=lambda: TensorDef())

    @property
    def shape(self):
        return self.output.shape

    @property
    def dtype(self):
        return self.output.dtype

    @property
    def nelems(self):
        return self.output.nelems

    @property
    def nbytes(self):
        return self.output.nbytes

    @property
    def dsize(self):
        return self.output.dsize


@dataclass
class Calculation(Operation):
    """Calculation node in computation graph."""

    function: Optional["LayerDef"] = None

    @property
    def nelems(self) -> int:
        return self.output.nelems

    @property
    def nbytes(self):
        return self.output.nelems * self.output.dsize

    @property
    def nbytes_weight(self) -> int:
        return self.function.nbytes_weight

    @property
    def nbytes_weight_grads(self) -> int:
        return self.function.nbytes_weight_grads

    @property
    def nbytes_activity(self) -> int:
        with hp.scope(**{"blueprinting.layerdef.inputs": self.inputs}):
            return self.function.nbytes_activity

    @property
    def flops_fw(self) -> int:
        with hp.scope(**{"blueprinting.layerdef.inputs": self.inputs}):
            return self.function.flops_fw

    @property
    def flops_bw(self) -> int:
        with hp.scope(**{"blueprinting.layerdef.inputs": self.inputs}):
            return self.function.flops_bw

    @property
    def memrw_fw(self) -> int:
        with hp.scope(**{"blueprinting.layerdef.inputs": self.inputs}):
            return self.function.memrw_fw

    @property
    def memrw_bw(self) -> int:
        with hp.scope(**{"blueprinting.layerdef.inputs": self.inputs}):
            return self.function.memrw_bw

    @property
    def time_fw(self) -> int:
        with hp.scope(**{"blueprinting.layerdef.inputs": self.inputs}):
            return self.function.time_fw

    @property
    def time_bw(self) -> int:
        with hp.scope(**{"blueprinting.layerdef.inputs": self.inputs}):
            return self.function.time_bw

    @property
    def memory_fw(self) -> int:
        with hp.scope(**{"blueprinting.layerdef.inputs": self.inputs}):
            return self.function.memory_fw

    @property
    def memory_bw(self) -> int:
        with hp.scope(**{"blueprinting.layerdef.inputs": self.inputs}):
            return self.function.memory_bw

    @property
    @hp.param("blueprinting.layerdef")
    def placement_weight(self, inputs: List[TensorDef] = None) -> Tuple[TensorDef, ...]:
        if inputs is None:
            inputs = []
        with hp.scope(**{"blueprinting.layerdef.inputs": self.inputs}):
            return self.function.placement_weight

    def subs(self, subs=None):
        """Substitute symbolic values in the output tensor."""
        if subs is None:
            subs = {}
        return self.output.subs(subs)

    def belike(self):
        """Return a TensorDef with the same shape and dtype as output."""
        return self.output.belike()

    def __or__(self, other):
        other(self)
