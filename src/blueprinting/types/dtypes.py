"""Data types for blueprinting."""

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "DType",
    "FloatPoint",
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "fp64",
    "float16",
    "bfloat16",
    "float32",
]


@dataclass
class FloatPoint:
    kind: str
    size: int


class DType(FloatPoint, Enum):
    """data types

    Examples
    --------
    >>> DType.fp8
    fp8
    >>> DType.fp16 == DType.float16
    True
    >>> DType("fp16")
    fp16
    >>> DType("float16")
    fp16
    >>> list(DType)
    [fp8, fp16, bf16, fp32, fp64]
    """

    fp8 = "fp8", 1
    fp16 = "fp16", 2
    bf16 = "bf16", 2
    fp32 = "fp32", 4
    fp64 = "fp64", 8
    float16 = "fp16", 2
    bfloat16 = "bf16", 2
    float32 = "fp32", 4

    def __repr__(self):
        return self.name

    @classmethod
    def _missing_(cls, value):
        for x in cls:
            if x.name == value:
                return x
        if value == "float16":
            return cls.fp16
        if value == "bfloat16":
            return cls.bf16
        if value == "float32":
            return cls.fp32
        return None


fp8, fp16, bf16, fp32, fp64 = list(DType)
float16, bfloat16, float32 = DType.float16, DType.bfloat16, DType.float32
