"""Blueprinting public package boundary.

Domain contracts live in explicit packages such as :mod:`blueprinting.workload`,
:mod:`blueprinting.mapping`, :mod:`blueprinting.system`, and
:mod:`blueprinting.synthesizer`.  The package root intentionally avoids broad
re-exports so importing Blueprinting does not initialize a legacy simulation
stack or hide domain ownership.
"""

from .__about__ import __version__

__all__ = ["__version__"]
