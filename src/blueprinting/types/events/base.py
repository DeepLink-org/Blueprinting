"""Base event classes for blueprinting."""

from typing import Any

__all__ = ["Event"]


class Event(dict):
    """Base event class.

    Examples
    --------
    >>> event = Event()
    >>> event.name = "event name"
    >>> event.name
    'event name'
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        return self.__setitem__(name, value)
