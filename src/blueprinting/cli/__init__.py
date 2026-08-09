"""Command-line entry points for supported Blueprinting applications."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from blueprinting import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blueprinting",
        description="Evidence-driven hardware architecture exploration",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("version", help="print the installed Blueprinting version")
    workbench = commands.add_parser("workbench", help="launch the NiceGUI architecture workbench")
    workbench.add_argument("--host", default="127.0.0.1", help="interface to bind")
    workbench.add_argument("--port", type=int, default=8080, help="TCP port")
    workbench.add_argument("--no-open", action="store_true", help="do not open a browser automatically")
    workbench.add_argument("--reload", action="store_true", help="reload when Python sources change")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch supported commands without importing UI code eagerly."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "workbench":
        from blueprinting.workbench.nicegui_app import run_workbench

        run_workbench(host=args.host, port=args.port, show=not args.no_open, reload=args.reload)
        return 0
    parser.print_help()
    return 0


__all__ = ["build_parser", "main"]
