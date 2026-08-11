"""Chrome Trace export for the workbench's portable dependency projection.

This module deliberately does not call the result a ``TimelineBundle``.  The
source is a PortablePlan task projection with analytical durations, not an
architecture-bound ConcretePlan or simulator trace.  Keeping that distinction
in the serialized metadata prevents a visualization export from being mistaken
for an executable schedule.
"""

from __future__ import annotations

import json
from typing import Any

from blueprinting.application import AnalysisReport

from .presentation import task_dependency_projection

TRACE_SCHEMA = "blueprinting.chrome-trace.portable-projection.v0"
TRACE_KIND = "portable_dependency_projection"


def portable_projection_trace(report: AnalysisReport) -> dict[str, Any]:
    """Return a deterministic Chrome Trace document for a portable task DAG."""

    rows = task_dependency_projection(report)
    by_id = {str(row["task_id"]): row for row in rows}
    phases = tuple(dict.fromkeys(str(row["phase"]) for row in rows))
    thread_ids = {phase: index + 1 for index, phase in enumerate(phases)}
    process_id = 1
    events: list[dict[str, Any]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": process_id,
            "tid": 0,
            "args": {"name": "PortablePlan dependency projection"},
        },
        {
            "name": "blueprinting.timeline",
            "ph": "M",
            "pid": process_id,
            "tid": 0,
            "args": {
                "schema": TRACE_SCHEMA,
                "timeline_kind": TRACE_KIND,
                "source_plan_digest": report.plan_digest,
                "request_digest": report.request_digest,
                "evidence_revision": report.evidence_revision,
                "executable": False,
                "scope": "one-local-tensor-parallel-block",
                "limitations": list(report.limitations),
            },
        },
    ]
    events.extend(
        {
            "name": "thread_name",
            "ph": "M",
            "pid": process_id,
            "tid": thread_id,
            "args": {"name": phase.replace("_", " ")},
        }
        for phase, thread_id in thread_ids.items()
    )

    tasks = {task.task_id: task for task in report.tasks}
    for row in rows:
        task_id = str(row["task_id"])
        task = tasks[task_id]
        events.append(
            {
                "name": str(row["operation"]),
                "cat": f"portable,{row['engine']},{row['phase']}",
                "ph": "X",
                "ts": _microseconds(row["start_seconds"]),
                "dur": _microseconds(row["duration_seconds"]),
                "pid": process_id,
                "tid": thread_ids[str(row["phase"])],
                "args": {
                    "task_id": task_id,
                    "source_layer": row["source_layer"],
                    "phase": row["phase"],
                    "engine": row["engine"],
                    "concurrency_group": row["concurrency_group"],
                    "dependency_count": row["dependencies"],
                    "operations": task.operations,
                    "read_bytes": task.read_bytes,
                    "write_bytes": task.write_bytes,
                    "message_bytes": task.message_bytes,
                    "evidence_provider": task.evidence_provider,
                    "evidence_revision": task.evidence_revision,
                    "evidence_method": task.evidence_method,
                    "timeline_kind": TRACE_KIND,
                    "executable": False,
                },
            }
        )

    for task in report.tasks:
        target = by_id[task.task_id]
        for dependency_id in task.dependencies:
            source = by_id.get(dependency_id)
            if source is None:
                continue
            flow_id = f"{dependency_id}->{task.task_id}"
            events.extend(
                (
                    {
                        "name": "dependency",
                        "cat": "portable.dependency",
                        "ph": "s",
                        "id": flow_id,
                        "ts": _microseconds(source["end_seconds"]),
                        "pid": process_id,
                        "tid": thread_ids[str(source["phase"])],
                        "args": {"from_task": dependency_id, "to_task": task.task_id},
                    },
                    {
                        "name": "dependency",
                        "cat": "portable.dependency",
                        "ph": "f",
                        "id": flow_id,
                        "bp": "e",
                        "ts": _microseconds(target["start_seconds"]),
                        "pid": process_id,
                        "tid": thread_ids[str(target["phase"])],
                        "args": {"from_task": dependency_id, "to_task": task.task_id},
                    },
                )
            )

    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "schema": TRACE_SCHEMA,
            "timeline_kind": TRACE_KIND,
            "source_plan_digest": report.plan_digest,
            "request_digest": report.request_digest,
            "evidence_revision": report.evidence_revision,
            "executable": False,
        },
    }


def portable_projection_trace_json(report: AnalysisReport, *, indent: int | None = None) -> str:
    """Serialize :func:`portable_projection_trace` without non-standard values."""

    return json.dumps(portable_projection_trace(report), ensure_ascii=False, indent=indent, allow_nan=False)


def perfetto_open_javascript(trace_json: str, *, title: str, filename: str) -> str:
    """Build a synchronous click handler that opens this trace in Perfetto.

    Perfetto's documented embedded-open protocol uses a PING/PONG handshake and
    transfers an ArrayBuffer.  A small Blob-backed bridge page keeps the popup
    creation synchronous with the user's click, avoiding common popup blockers.
    """

    bridge_html = _perfetto_bridge_html(trace_json, title=title, filename=filename)
    encoded_html = json.dumps(bridge_html, ensure_ascii=True)
    return (
        "() => {"
        f"const html = {encoded_html};"
        "const blob = new Blob([html], {type: 'text/html'});"
        "const url = URL.createObjectURL(blob);"
        "const popup = window.open(url, '_blank');"
        "if (!popup) { alert('浏览器阻止了弹窗，请允许本站打开新窗口。'); }"
        "setTimeout(() => URL.revokeObjectURL(url), 60000);"
        "}"
    )


def _perfetto_bridge_html(trace_json: str, *, title: str, filename: str) -> str:
    trace_literal = _embedded_javascript_string(trace_json)
    title_literal = _embedded_javascript_string(title)
    filename_literal = _embedded_javascript_string(filename)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Blueprinting → Perfetto</title>
<style>html,body,iframe{{width:100%;height:100%;margin:0;border:0}}#status{{font:14px system-ui;padding:18px}}</style>
</head><body><div id="status">正在向 Perfetto 发送 Blueprinting timeline…</div>
<iframe id="perfetto" hidden src="https://ui.perfetto.dev/#!/"></iframe>
<script>
const traceText = {trace_literal};
const traceTitle = {title_literal};
const traceFilename = {filename_literal};
const iframe = document.getElementById('perfetto');
const status = document.getElementById('status');
let sent = false;
window.addEventListener('message', event => {{
  if (event.origin !== 'https://ui.perfetto.dev' || event.source !== iframe.contentWindow || event.data !== 'PONG' || sent) return;
  sent = true;
  const buffer = new TextEncoder().encode(traceText).buffer;
  iframe.contentWindow.postMessage({{perfetto: {{buffer, title: traceTitle, fileName: traceFilename}}}}, 'https://ui.perfetto.dev');
  status.remove(); iframe.hidden = false;
}});
iframe.addEventListener('load', () => {{
  let attempts = 0;
  const timer = setInterval(() => {{
    if (sent || attempts++ >= 20) {{ clearInterval(timer); return; }}
    iframe.contentWindow.postMessage('PING', 'https://ui.perfetto.dev');
  }}, 250);
}});
</script></body></html>"""


def _embedded_javascript_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _microseconds(seconds: Any) -> float:
    return float(seconds) * 1_000_000.0


__all__ = [
    "TRACE_KIND",
    "TRACE_SCHEMA",
    "perfetto_open_javascript",
    "portable_projection_trace",
    "portable_projection_trace_json",
]
