from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from test_presentation import _analysis_report

from blueprinting.workbench.chrome_trace import (
    TRACE_KIND,
    perfetto_open_javascript,
    portable_projection_trace,
    portable_projection_trace_json,
)


def test_portable_projection_exports_tasks_metadata_and_dependency_flows() -> None:
    report = _analysis_report()
    document = portable_projection_trace(report)
    events = document["traceEvents"]

    task_events = [event for event in events if event.get("ph") == "X"]
    flow_starts = [event for event in events if event.get("ph") == "s"]
    flow_ends = [event for event in events if event.get("ph") == "f"]

    assert len(task_events) == len(report.tasks)
    assert len(flow_starts) == sum(len(task.dependencies) for task in report.tasks)
    assert {event["id"] for event in flow_starts} == {event["id"] for event in flow_ends}
    assert document["metadata"] == {
        "schema": "blueprinting.chrome-trace.portable-projection.v1",
        "timeline_kind": TRACE_KIND,
        "source_plan_digest": report.plan_digest,
        "request_digest": report.request_digest,
        "evidence_revision": report.evidence_revision,
        "executable": False,
    }
    assert all(event["args"]["executable"] is False for event in task_events)


def test_portable_projection_trace_json_is_deterministic_and_standard_json() -> None:
    report = _analysis_report()

    first = portable_projection_trace_json(report)
    second = portable_projection_trace_json(report)

    assert first == second
    assert json.loads(first)["displayTimeUnit"] == "ms"


def test_perfetto_click_handler_uses_array_buffer_handshake() -> None:
    script = perfetto_open_javascript(
        '{"traceEvents":[]}',
        title="Blueprinting </script>",
        filename="trace.json",
    )

    assert "window.open" in script
    assert "ui.perfetto.dev" in script
    assert "TextEncoder" in script
    assert "PING" in script and "PONG" in script
    assert "Blueprinting </script>" not in script


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for JavaScript execution test")
def test_perfetto_click_handler_executes_with_browser_primitives() -> None:
    script = perfetto_open_javascript('{"traceEvents":[]}', title="Blueprinting", filename="trace.json")
    harness = """
global.Blob = class Blob { constructor(parts) { this.parts = parts; } };
global.URL = {createObjectURL: () => 'blob:test', revokeObjectURL: () => {}};
let openCount = 0;
global.window = {open: () => { openCount += 1; return {}; }};
global.alert = () => {};
global.setTimeout = callback => callback();
const handler = new Function('return (' + process.argv[1] + ');')();
if (openCount !== 0) throw new Error('handler opened Perfetto during registration');
handler();
if (openCount !== 1) throw new Error('handler did not open Perfetto on click');
process.stdout.write('opened');
"""

    completed = subprocess.run(
        ["node", "-e", harness, script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == "opened"
