"""Base class for compiler passes."""

from abc import ABC, abstractmethod
from typing import TypeVar, Union

from ..graph import GraphIR
from ..schedule import ScheduleIR
from ..result import SimulationResult

# Type variable for IR types
IR = TypeVar("IR", GraphIR, ScheduleIR, SimulationResult)


class Pass(ABC):
    """Abstract base class for compiler passes.
    
    A Pass transforms an IR (Intermediate Representation) into
    another IR or a final result. Passes can be chained together
    to form a compilation pipeline.
    
    Example:
        class MyPass(Pass):
            def run(self, ir: GraphIR) -> GraphIR:
                # Transform the graph
                for node in ir.nodes.values():
                    node.attrs["processed"] = True
                return ir
    """
    
    @abstractmethod
    def run(self, ir: IR) -> IR:
        """Execute the pass on the given IR.
        
        Args:
            ir: Input IR (GraphIR, ScheduleIR, etc.)
            
        Returns:
            Transformed IR or result
        """
        pass
    
    @property
    def name(self) -> str:
        """Name of this pass."""
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        return f"{self.name}()"


class IdentityPass(Pass):
    """A pass that does nothing (for testing)."""
    
    def run(self, ir: IR) -> IR:
        return ir


class ComposedPass(Pass):
    """A pass that composes multiple passes."""
    
    def __init__(self, *passes: Pass):
        self.passes = list(passes)
    
    def run(self, ir: IR) -> IR:
        result = ir
        for p in self.passes:
            result = p.run(result)
        return result
    
    @property
    def name(self) -> str:
        names = [p.name for p in self.passes]
        return f"ComposedPass({', '.join(names)})"
