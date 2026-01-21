"""Model mixins for blueprinting.

This module provides backward compatibility. New code should use the model/ submodule directly.
"""

from .model import ModelFlops, ModelParams

# Backward compatibility aliases
ModelParamsMixin = ModelParams
ModelFlopsMixin = ModelFlops


class ModelMixin:
    """Empty mixin for backward compatibility."""

    def __init__(self, cfg) -> None:
        pass


__all__ = ["ModelMixin", "ModelParamsMixin", "ModelFlopsMixin"]
