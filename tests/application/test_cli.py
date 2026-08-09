from __future__ import annotations

import blueprinting
from blueprinting.cli import build_parser, main


def test_package_root_exposes_only_version_metadata() -> None:
    assert blueprinting.__all__ == ["__version__"]
    assert isinstance(blueprinting.__version__, str)
    assert blueprinting.__version__
    assert not hasattr(blueprinting, "Analyzer")
    assert not hasattr(blueprinting, "System")


def test_cli_version_command_has_no_application_side_effects(capsys) -> None:
    assert main(["version"]) == 0

    assert capsys.readouterr().out.strip() == blueprinting.__version__


def test_cli_parses_workbench_binding_without_importing_the_ui() -> None:
    arguments = build_parser().parse_args(["workbench", "--host", "0.0.0.0", "--port", "9000", "--no-open", "--reload"])

    assert arguments.command == "workbench"
    assert arguments.host == "0.0.0.0"
    assert arguments.port == 9000
    assert arguments.no_open is True
    assert arguments.reload is True
