# ADR-0004: Use Domain-Owned Semantic Wire Identities

- Date: 2026-08-11
- Status: Accepted
- Scope: canonical record/enum tags, digest domains, compatibility policy
- Supersedes: the codec-tag preservation clauses in ADR-0001 and ADR-0002

## Context

Canonical tags still reflected an obsolete implementation package. That name
appeared throughout IR declarations, workload and system contracts, analysis
records, tests, and generated JSON even though it no longer described a
Blueprinting domain. Treating the prefix as forever opaque made the source
harder to read and preserved accidental architecture in every new artifact.

## Decision

Canonical identities now state semantic ownership directly:

| Namespace | Owner |
|---|---|
| `blueprinting.ir.*` | Shared primitives and the five canonical IR stages |
| `blueprinting.binding.*` | Explicit derivation bindings |
| `blueprinting.workload.*` | Target-neutral workload contracts |
| `blueprinting.mapping.*` | Logical strategies and deployment mapping |
| `blueprinting.system.*` | Chip, memory, interconnect, and system profiles |
| `blueprinting.analysis.*` | Evidence and rebuildable analysis records |
| `blueprinting.synthesis.*` | Derivation-session state |
| `blueprinting.expression.*` | Typed scalar expressions |

Names use kebab-case components without independent version counters. ADT
constructors keep short local tags; the family expands them under its semantic
namespace. Compatibility is governed by the owning IR root schema, not by a
second version embedded in every nested type name.

This is an intentional hard wire-format cut. The runtime codec does not
register obsolete tag aliases or field aliases. Existing persisted artifacts
must be regenerated from their authoritative workload, mapping, system, and
evidence inputs. Canonical digests change by design; numerical facts and
derivation semantics do not.

Schema migrations remain available for future graduated root-schema changes.
The current production registry is empty because no pre-graduation intermediate
state is treated as released compatibility history.

## Consequences

- IR declarations explain domain ownership without historical context.
- Newly encoded values contain only semantic `blueprinting.*` identities.
- Old serialized snapshots fail closed instead of silently entering current
  derivations through aliases.
- Golden digests and generated experiment artifacts must be regenerated in the
  same change.
- External consumers must treat this release as a wire-format boundary.

## Verification

- A source gate rejects canonical decorators whose tag does not begin with
  `blueprinting.` and rejects reintroduction of the abandoned prefix.
- Canonical round-trip, schema migration, determinism, and baseline tests run
  against regenerated semantic-namespace artifacts.
- Ruff, mypy, full pytest, bilingual documentation, and wheel-contract checks
  remain release gates.
