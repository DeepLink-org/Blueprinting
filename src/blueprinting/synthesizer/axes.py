"""Typed binding axes shared by expressions, sessions, and passes."""

from enum import Enum

from blueprinting.schema.authoring import enum


@enum("blueprinting.binding.axis")
class BindingAxis(Enum):
    WORKLOAD = "workload"
    STRATEGY = "strategy"
    TARGET = "target"
    DEPLOYMENT = "deployment"
    CALIBRATION = "calibration"
