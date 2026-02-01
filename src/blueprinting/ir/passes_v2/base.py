"""Base class for v2 passes."""

from abc import ABC, abstractmethod
from typing import TypeVar, Union

from ..types import GraphIR, ScheduleIR, TimelineIR, SimulationResult

# Type variable for IR types
IR = TypeVar("IR", GraphIR, ScheduleIR, TimelineIR, SimulationResult)


class Pass(ABC):
    """Abstract base class for compiler passes.
    
    A Pass transforms one IR type into another (or the same type).
    
    Pass chain:
        GraphIR → ScheduleIR → TimelineIR → SimulationResult
    """
    
    @property
    def name(self) -> str:
        """Pass name."""
        return self.__class__.__name__
    
    @abstractmethod
    def run(self, ir: IR) -> IR:
        """Execute the pass."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.name}()"
    
    def __call__(self, ir: IR) -> IR:
        """Allow pass to be called as a function."""
        return self.run(ir)


class Pipeline:
    """A pipeline of passes.
    
    Usage:
        pipeline = Pipeline([
            PrintGraphPass(),
            ExpandPass(),
            PrintSchedulePass(),
        ])
        result = pipeline.run(graph_ir)
    """
    
    def __init__(self, passes: list):
        self.passes = passes
    
    def run(self, ir):
        """Run all passes in sequence."""
        result = ir
        for p in self.passes:
            result = p.run(result)
        return result
    
    def __repr__(self) -> str:
        names = [p.name for p in self.passes]
        return f"Pipeline({' → '.join(names)})"
