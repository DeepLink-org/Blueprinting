# Floating-Point Numerical Analysis

Blueprinting's interactive floating-point analysis is a **numerical-fact exploration surface** in the workspace. It decodes IEEE-style binary floating-point formats, exposes dynamic range and representable values, and quantifies quantization and overflow/underflow risk for candidate datatypes. It is a presentation analysis view — it predicts neither execution time nor hardware behavior, and it does not define workload semantics.

## Scope and motivation

Datatype is a first-class workload fact: `WorkloadFacts` and plan buffers carry exact datatype and bytes-per-element, and Transformer derivation treats precision as a semantic input (see [workload model](../modeling/workload.md)). Before committing a blueprint or mapping to a datatype (bf16 vs fp16 vs fp8), an architect needs numerical questions answered:

- what the format actually represents (dynamic range, subnormal region, Inf/NaN);
- how large the nearest-value quantization error is at the magnitudes the workload produces;
- which operations (add, subtract, multiply, divide) risk underflow or overflow at the format's limits.

The floating-point panel answers these questions interactively, for standard formats and arbitrary custom layouts.

## Analysis surface

The panel exposes five connected views over one selected format.

### Format layout comparison

Standard presets — `fp32`, `tf32`, `bf16`, `fp16`, `fp8(E5M2)`, `fp8(E4M3)`, `fp4(E2M1)` — plus a custom format defined by sign bit (optional), exponent bits (`2..8`), and mantissa bits (`0..23`). The layout chart compares sign/exponent/mantissa widths across formats.

### Bit-level decoding

A bit editor decodes one concrete pattern with IEEE zero/subnormal/special handling: all-zero exponent with zero fraction is zero; all-zero exponent with nonzero fraction is subnormal; all-ones exponent with zero fraction is infinity; all-ones exponent with nonzero fraction is NaN. The decode shows category, raw exponent, significand, and value.

### Dynamic range and representable values

For the selected format the panel reports bias, min normal, min subnormal, and max finite. The representable-values view enumerates finite values within the observation window and marks normal (blue) and subnormal (red) regions.

### Quantization error

The nearest-value quantization error curve samples a dense input range and reports the absolute error to the nearest representable value, exposing the format's precision at the magnitudes of interest.

### Operation range impact

A Cartesian sample over up to 64 downsampled representable values runs `A + B`, `A − B`, `A × B`, and `A ÷ B`, and counts results that stay normal, become subnormal, underflow to zero, or overflow to Inf/NaN. This is numerical range analysis, not hardware execution time.

## Place in the analysis architecture

- The analysis functions are pure and deterministic; they live in `src/blueprinting/workbench/float_analysis.py`, with value enumeration in `src/blueprinting/fp/`.
- The panel does not consume `ModelIR`, `PortablePlanIR`, or any canonical IR. It is an independent interactive surface, not a derived view of a canonical plan.
- It produces no cost estimates, so it is not a cost provider and never enters the evidence-resolution path.
- Interactive enumeration is intentionally bounded to formats with exponent + mantissa bits ≤ 12 to keep the UI responsive.

## Current boundary and next steps

**Implemented:** the pure analysis functions, the workbench panel, and unit tests covering dynamic range, bit decoding, enumeration limits, and operation-impact completeness.

**Not yet implemented:** coupling the analysis to workload datatype bindings, deriving per-operation quantization/overflow guardrails for a training or inference plan, and fp8 scaling-policy analysis. A future slice could expose numerical facts as an analysis addressed by a plan's datatype digest, without changing workload semantics.

The workspace that hosts the panel is described in [Exploration Workspace](../exploration/workspace.md); project-wide implementation status is tracked in [Implementation Status](../project/status.md).
