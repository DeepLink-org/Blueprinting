"""Typed binding axes shared by expressions, sessions, and passes."""

from enum import Enum

from blueprinting.schema.codec import enum_type


@enum_type("blueprinting.binding.axis")
class BindingAxis(Enum):
    WORKLOAD = "workload"
    STRATEGY = "strategy"
    TARGET = "target"
    DEPLOYMENT = "deployment"
    CALIBRATION = "calibration"
