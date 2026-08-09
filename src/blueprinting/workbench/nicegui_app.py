"""Executable entry point for the Blueprinting NiceGUI workbench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import suppress

from nicegui import ui

from .nicegui_ui import workbench_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Blueprinting architecture workbench")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="TCP port (default: 8080)")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser automatically")
    parser.add_argument("--reload", action="store_true", help="reload the server when Python sources change")
    return parser


def run_workbench(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    show: bool = True,
    reload: bool = False,
) -> None:
    """Run the primary Blueprinting UI; the legacy Streamlit app stays separate."""

    ui.run(
        workbench_root,
        host=host,
        port=port,
        title="Blueprinting · Architecture Workbench",
        favicon="🧭",
        dark=False,
        language="zh-CN",
        show=show,
        reload=reload,
        show_welcome_message=False,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    with suppress(KeyboardInterrupt):
        run_workbench(host=args.host, port=args.port, show=not args.no_open, reload=args.reload)


if __name__ in {"__main__", "__mp_main__"}:
    main()
