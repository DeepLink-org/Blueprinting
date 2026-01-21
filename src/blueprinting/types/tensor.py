"""Tensor definitions for blueprinting."""

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Tuple, Union

from sympy import Expr

from .dtypes import DType, fp32

__all__ = [
    "TensorLike",
    "TensorDef",
]


class TensorLike:
    """Tensor protocol mixin."""

    @property
    def nelems(self):
        return reduce(mul, self.shape, 1)

    @property
    def nbytes(self):
        return self.dsize * self.nelems

    @property
    def dsize(self):
        return self.dtype.size

    @property
    def T(self):
        return TensorDef([x for x in reversed(self.shape)], self.dtype)

    def belike(self):
        return TensorDef([x for x in self.shape], self.dtype)

    def __repr__(self) -> str:
        return f"{self.dtype}{self.shape}"

    def subs(self, subs={}) -> "TensorDef":
        return TensorDef(
            [int(x.subs(subs)) if isinstance(x, Expr) else x for x in self.shape],
            dtype=self.dtype,
        )


@dataclass
class TensorDef(TensorLike):
    """TensorDef

    Examples
    --------
    >>> TensorDef([1, 2, 3], dtype=DType.fp32)
    fp32[1, 2, 3]
    """

    shape: Tuple[Union[int, Expr], ...] = tuple()
    dtype: DType = field(default_factory=lambda: fp32)

    def __repr__(self) -> str:
        return f"{repr(self.dtype)}{self.shape}"
