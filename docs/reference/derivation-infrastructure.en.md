# Derivation infrastructure API

The infrastructure separates three concerns:

- schema authoring creates immutable, codec-visible records and closed ADTs;
- pass authoring extracts static contracts from annotations without wrapping execution;
- the transaction runner verifies inputs, outputs, lineage rules, deterministic replay, analyses, and checkpoints before commit.

## Schema authoring

::: blueprinting.schema.authoring
    options:
      members:
        - record
        - adt
        - variant
        - adt_manifest
      show_root_heading: false
      show_root_toc_entry: false

## Pass authoring

::: blueprinting.synthesizer.passes.authoring
    options:
      members:
        - derivation
        - relation
        - claim
      show_root_heading: false
      show_root_toc_entry: false

## Transaction and verification types

::: blueprinting.synthesizer.passes.base
    options:
      members:
        - PassRule
        - TransitionRelation
        - TransitionReport
        - TransitionVerifier
        - PassContract
        - PassContext
        - PassResult
        - DerivationPass
        - PassPipeline
        - PassRecord
        - PassCheckpoint
        - PassManager
      show_root_heading: false
      show_root_toc_entry: false
