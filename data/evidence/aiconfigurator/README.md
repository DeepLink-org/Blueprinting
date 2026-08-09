# AIConfigurator evidence bundle

This directory is an external performance-data snapshot, not a Python package
and not a Blueprinting system contract.  Consumers must load individual files
through an explicit evidence importer and record the selected file digest,
runtime/backend revision, hardware identity, and measurement protocol.

The imported snapshot does not currently include one repository-level source
revision manifest.  Treat it as exploratory evidence; do not use it as a frozen
regression oracle until provenance and license metadata are pinned for the
whole bundle.  The files retain their original SPDX headers where supplied.

The directory is intentionally excluded from default wheel and source
distribution artifacts because it is large and optional.
