"""Event parsing utilities for blueprinting."""

import ast
from dataclasses import dataclass
from typing import Any, List

from .base import Event

__all__ = ["parse", "parse_tree"]


@dataclass
class TensorDef:
    """Tensor definition for parsing.

    Note: This is used internally by the parser for representing tensor shapes
    when parsing event logs.
    """

    shape: tuple = ()
    dtype: Any = None

    def __repr__(self):
        return f"TensorDef({self.shape}, {self.dtype})"


def parse(line: str) -> Event:
    """Parse a single event line.

    Args:
        line: Event line string

    Returns:
        Parsed Event object
    """
    expr = ast.parse(line, mode="eval")
    name = expr.body.func.id
    args = [eval(compile(ast.Expression(body=arg), filename="", mode="eval")) for arg in expr.body.args]
    event = Event()
    event.name = name
    event.module = args[0]
    if name == "SpanStartEvent":
        event.id = args[1]
    elif name == "SpanEndEvent":
        event.id = args[1]
        event.duration = args[2]
    else:
        event.inputs = args[1]
        event.params = args[2] if len(args) > 2 else {}
    return event


def parse_tree(lines: List[str]) -> List[Event]:
    """Parse event lines into a tree structure.

    Args:
        lines: List of event line strings

    Returns:
        List of Event trees
    """
    tree = []

    span_stack = []
    curr_span = None

    event_stack = []
    for line in lines:
        event = parse(line)
        if event.name == "SpanStartEvent":
            new_span = Event()
            if curr_span is not None:
                curr_span.children.append(new_span)
            curr_span = new_span
            curr_span.name = "Span"
            curr_span.module = event.module
            curr_span.id = event.id
            curr_span.duration = None
            curr_span.children = []
            tree.append(curr_span)
            span_stack.append(curr_span)
        if event.name == "SpanEndEvent":
            curr_span.duration = event.duration
            curr_span.children = curr_span.children
            curr_span = span_stack.pop()

        if event.name == "ForwardStartEvent":
            if curr_span is not None:
                curr_span.children.append(event)
                event_stack.append(event)

        if event.name == "ForwardEndEvent":
            if len(event_stack) > 0:
                curr_event = event_stack.pop()
                curr_event.outputs = event.inputs[0]

    return tree
