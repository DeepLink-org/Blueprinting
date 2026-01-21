"""Model communication mixin for blueprinting.

This module provides backward compatibility. New code should use the model/ submodule directly.
"""

from .model import ModelComm

# Backward compatibility alias
ModelCommMixin = ModelComm

__all__ = ["ModelCommMixin"]
