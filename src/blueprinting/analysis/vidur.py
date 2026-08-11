"""Strict adapter for user-supplied Vidur profiling CSV files.

The adapter intentionally performs exact-key lookup only.  It does not copy
Vidur datasets into Blueprinting and it never silently interpolates across a
different model, tensor-parallel degree, batch size or context length.
"""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from pathlib import Path

from blueprinting.schema.codec import content_digest
from blueprinting.schema.frozen import FrozenDict

from ..synthesizer.bindings import InferencePhase
from .cost.database import EvidenceProvenance, PerformanceDatabase, PerformanceRecord
from .cost.protocol import CostSubject, EstimateMethod
from .inference_evidence import InferenceEvidenceQuery, InferenceEvidenceResult

_COMPUTE_COLUMNS = {
    "input_layernorm": "time_stats.input_layernorm.median",
    "attention_pre_projection": "time_stats.attn_pre_proj.median",
    "attention_rope": "time_stats.attn_rope.median",
    "attention_post_projection": "time_stats.attn_post_proj.median",
    "post_attention_layernorm": "time_stats.post_attention_layernorm.median",
    "mlp_up_projection": "time_stats.mlp_up_proj.median",
    "mlp_activation": "time_stats.mlp_act.median",
    "mlp_down_projection": "time_stats.mlp_down_proj.median",
    "residual_add": "time_stats.add.median",
}

_SOURCE_LAYERS = {
    "input_layernorm": "attention.input_norm",
    "attention_pre_projection": "attention.qkv",
    "attention_rope": "attention.rope",
    "attention_post_projection": "attention.output",
    "post_attention_layernorm": "mlp.input_norm",
    "mlp_up_projection": "mlp.up",
    "mlp_activation": "mlp.activation",
    "mlp_down_projection": "mlp.down",
    "residual_add": "mlp.residual",
}

_GEMM_PRIMITIVES = frozenset(
    {
        "attention_pre_projection",
        "attention_post_projection",
        "mlp_up_projection",
        "mlp_down_projection",
    }
)


def _read_rows(path: Path, *, required: frozenset[str], timing_columns: frozenset[str]) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Vidur profile {path} has no CSV header")
        fields = frozenset(reader.fieldnames)
        missing = required - fields
        if missing:
            raise ValueError(f"Vidur profile {path} is missing columns: {', '.join(sorted(missing))}")
        if fields.isdisjoint(timing_columns):
            raise ValueError(f"Vidur profile {path} has no supported timing column")
        rows = tuple(dict(row) for row in reader)
        if not rows:
            raise ValueError(f"Vidur profile {path} has no data rows")
        return rows


def _integer(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        numeric = float(value)
    except ValueError:
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _boolean(row: dict[str, str], key: str) -> bool | None:
    value = row.get(key, "").strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    return None


def _milliseconds(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) and result >= 0 else None


class VidurProfileBaseline:
    """Exact-match comparison baseline for Vidur's public CSV schema."""

    def __init__(
        self,
        *,
        attention_rows: tuple[dict[str, str], ...],
        compute_rows: tuple[dict[str, str], ...],
        model_name: str,
        hardware_name: str,
        attention_backend: str,
        block_size: int,
        datatype: str,
        source_revision: str,
        data_digest: str,
    ) -> None:
        if any(
            not value
            for value in (model_name, hardware_name, attention_backend, datatype, source_revision, data_digest)
        ):
            raise ValueError("Vidur baseline identity fields must not be empty")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("Vidur block_size must be a positive integer")
        self._attention_rows = attention_rows
        self._compute_rows = compute_rows
        self._model_name = model_name
        self._hardware_name = hardware_name
        self._attention_backend = attention_backend
        self._block_size = block_size
        self._datatype = datatype
        self._revision = content_digest(
            FrozenDict(
                {
                    "adapter": "blueprinting-vidur-baseline-v0",
                    "upstream_revision": source_revision,
                    "data_digest": data_digest,
                    "model_name": model_name,
                    "hardware_name": hardware_name,
                    "attention_backend": attention_backend,
                    "block_size": block_size,
                    "datatype": datatype,
                }
            ),
            "inference-baseline",
        )

    @classmethod
    def from_csv(
        cls,
        *,
        attention_csv: str | Path,
        compute_csv: str | Path,
        model_name: str,
        hardware_name: str,
        attention_backend: str,
        block_size: int,
        source_revision: str,
        datatype: str = "float16",
    ) -> VidurProfileBaseline:
        """Load baseline profiles bound to an explicit upstream revision."""

        attention_path = Path(attention_csv)
        compute_path = Path(compute_csv)
        digester = hashlib.sha256()
        for path in (attention_path, compute_path):
            payload = path.read_bytes()
            digester.update(path.name.encode("utf-8"))
            digester.update(len(payload).to_bytes(8, "big"))
            digester.update(payload)
        return cls(
            attention_rows=_read_rows(
                attention_path,
                required=frozenset(
                    {
                        "n_embd",
                        "n_q_head",
                        "n_kv_head",
                        "num_tensor_parallel_workers",
                        "batch_size",
                        "prefill_chunk_size",
                        "kv_cache_size",
                        "is_prefill",
                        "attention_backend",
                        "block_size",
                        "max_model_len",
                    }
                ),
                timing_columns=frozenset(
                    {
                        "time_stats.attn_prefill.median",
                        "time_stats.attn_decode.median",
                        "time_stats.attn_kv_cache_save.median",
                    }
                ),
            ),
            compute_rows=_read_rows(
                compute_path,
                required=frozenset(
                    {
                        "n_embd",
                        "n_expanded_embd",
                        "n_head",
                        "n_kv_head",
                        "num_tensor_parallel_workers",
                        "num_tokens",
                        "use_gated_mlp",
                    }
                ),
                timing_columns=frozenset(_COMPUTE_COLUMNS.values()),
            ),
            model_name=model_name,
            hardware_name=hardware_name,
            attention_backend=attention_backend,
            block_size=block_size,
            datatype=datatype,
            source_revision=source_revision,
            data_digest=digester.hexdigest(),
        )

    @property
    def revision(self) -> str:
        return self._revision

    def lookup(self, query: InferenceEvidenceQuery) -> InferenceEvidenceResult | None:
        if (
            query.model_name != self._model_name
            or query.hardware_name != self._hardware_name
            or query.datatype != self._datatype
        ):
            return None
        # Vidur's block model contributes one ``add_time`` after the MLP.  The
        # attention residual in Blueprinting remains explicit work, but it has
        # no independently comparable public Vidur component record.
        if query.primitive == "residual_add" and query.source_layer != "mlp.residual":
            return None
        if query.primitive in {"attention_core", "attention_kv_cache_save"}:
            values = self._attention_values(query)
        else:
            values = self._compute_values(query)
        if not values:
            return None
        return InferenceEvidenceResult(
            seconds=statistics.median(values) * 1e-3,
            provider="vidur-profile-baseline",
            revision=self.revision,
            match=f"exact-median-{len(values)}",
        )

    def _attention_values(self, query: InferenceEvidenceQuery) -> tuple[float, ...]:
        is_prefill = query.phase is InferencePhase.PREFILL
        timing_column = (
            "time_stats.attn_kv_cache_save.median"
            if query.primitive == "attention_kv_cache_save"
            else ("time_stats.attn_prefill.median" if is_prefill else "time_stats.attn_decode.median")
        )
        expected_prefill_chunk = query.query_tokens if is_prefill else 0
        # Vidur records the cache length before the current decode token is
        # appended, while Blueprinting context_tokens is the number of keys
        # visible to attention after that append.
        expected_kv_cache = 0 if is_prefill else query.context_tokens - 1
        values = []
        for row in self._attention_rows:
            matches = (
                _integer(row, "n_embd") == query.hidden_size
                and _integer(row, "n_q_head") == query.attention_heads
                and _integer(row, "n_kv_head") == query.attention_heads
                and _integer(row, "num_tensor_parallel_workers") == query.tensor_parallel
                and _integer(row, "block_size") == self._block_size
                and _integer(row, "max_model_len") == query.model_sequence_length
                and _integer(row, "batch_size") == query.batch_size
                and _integer(row, "prefill_chunk_size") == expected_prefill_chunk
                and _integer(row, "kv_cache_size") == expected_kv_cache
                and _boolean(row, "is_prefill") is is_prefill
                and row.get("attention_backend") == self._attention_backend
            )
            value = _milliseconds(row, timing_column)
            if matches and value is not None:
                values.append(value)
        return tuple(values)

    def _compute_values(self, query: InferenceEvidenceQuery) -> tuple[float, ...]:
        timing_column = _COMPUTE_COLUMNS.get(query.primitive)
        if timing_column is None:
            return ()
        num_tokens = query.batch_size * query.query_tokens
        values = []
        for row in self._compute_rows:
            matches = (
                _integer(row, "n_embd") == query.hidden_size
                and _integer(row, "n_expanded_embd") == query.feedforward_size
                and _integer(row, "n_head") == query.attention_heads
                and _integer(row, "n_kv_head") == query.attention_heads
                and _integer(row, "num_tensor_parallel_workers") == query.tensor_parallel
                and _integer(row, "num_tokens") == num_tokens
                and _boolean(row, "use_gated_mlp") is False
            )
            value = _milliseconds(row, timing_column)
            if matches and value is not None:
                values.append(value)
        return tuple(values)


def _strict_integer(row: dict[str, str], key: str, row_number: int, table: str) -> int:
    value = _integer(row, key)
    if value is None:
        raise ValueError(f"Vidur {table} row {row_number} has invalid {key!r}")
    return value


def _strict_boolean(row: dict[str, str], key: str, row_number: int, table: str) -> bool:
    value = _boolean(row, key)
    if value is None:
        raise ValueError(f"Vidur {table} row {row_number} has invalid {key!r}")
    return value


class VidurProfileImporter:
    """Explicitly promote Vidur profile rows into a cost evidence database.

    This is deliberately separate from :class:`VidurProfileBaseline`.  Calling
    the importer is the policy decision that makes user-supplied profile data
    admissible to a ``CostResolver``; baseline lookup remains post-hoc only.
    """

    IMPORTER_REVISION = "blueprinting-vidur-profile-v0"

    @classmethod
    def from_csv(
        cls,
        *,
        attention_csv: str | Path,
        compute_csv: str | Path,
        model_name: str,
        hardware_name: str,
        source_revision: str,
        datatype: str = "float16",
        database_name: str | None = None,
    ) -> PerformanceDatabase:
        for name, value in (
            ("model_name", model_name),
            ("hardware_name", hardware_name),
            ("source_revision", source_revision),
            ("datatype", datatype),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        attention_path = Path(attention_csv)
        compute_path = Path(compute_csv)
        digester = hashlib.sha256()
        for path in (attention_path, compute_path):
            payload = path.read_bytes()
            digester.update(path.name.encode("utf-8"))
            digester.update(len(payload).to_bytes(8, "big"))
            digester.update(payload)
        data_digest = digester.hexdigest()
        attention_rows = _read_rows(
            attention_path,
            required=frozenset(
                {
                    "n_embd",
                    "n_q_head",
                    "n_kv_head",
                    "num_tensor_parallel_workers",
                    "batch_size",
                    "prefill_chunk_size",
                    "kv_cache_size",
                    "is_prefill",
                    "attention_backend",
                    "block_size",
                    "max_model_len",
                }
            ),
            timing_columns=frozenset(
                {
                    "time_stats.attn_prefill.median",
                    "time_stats.attn_decode.median",
                    "time_stats.attn_kv_cache_save.median",
                }
            ),
        )
        compute_rows = _read_rows(
            compute_path,
            required=frozenset(
                {
                    "n_embd",
                    "n_expanded_embd",
                    "n_head",
                    "n_kv_head",
                    "num_tensor_parallel_workers",
                    "num_tokens",
                    "use_gated_mlp",
                }
            ),
            timing_columns=frozenset(_COMPUTE_COLUMNS.values()),
        )
        provenance = EvidenceProvenance(
            source="vidur-profile",
            source_revision=source_revision,
            importer=cls.IMPORTER_REVISION,
            data_digest=data_digest,
            method=EstimateMethod.MEASURED,
            metadata=FrozenDict(
                {
                    "attention_file": attention_path.name,
                    "compute_file": compute_path.name,
                }
            ),
        )
        records = [
            *cls._attention_records(
                attention_rows,
                model_name=model_name,
                hardware_name=hardware_name,
                datatype=datatype,
                provenance=provenance,
                data_digest=data_digest,
            ),
            *cls._compute_records(
                compute_rows,
                model_name=model_name,
                hardware_name=hardware_name,
                datatype=datatype,
                provenance=provenance,
                data_digest=data_digest,
            ),
        ]
        return PerformanceDatabase(
            name=database_name or f"vidur-{model_name}-{hardware_name}",
            records=tuple(records),
            metadata=FrozenDict(
                {
                    "source": "vidur-profile",
                    "source_revision": source_revision,
                    "data_digest": data_digest,
                    "importer": cls.IMPORTER_REVISION,
                }
            ),
        )

    @classmethod
    def _record_id(cls, data_digest: str, table: str, row_number: int, metric: str) -> str:
        return content_digest(
            FrozenDict(
                {
                    "importer": cls.IMPORTER_REVISION,
                    "data_digest": data_digest,
                    "table": table,
                    "row": row_number,
                    "metric": metric,
                }
            ),
            "performance-record-id",
        )

    @classmethod
    def _attention_records(
        cls,
        rows: tuple[dict[str, str], ...],
        *,
        model_name: str,
        hardware_name: str,
        datatype: str,
        provenance: EvidenceProvenance,
        data_digest: str,
    ) -> tuple[PerformanceRecord, ...]:
        records = []
        for row_number, row in enumerate(rows, start=1):
            prefill = _strict_boolean(row, "is_prefill", row_number, "attention")
            batch_size = _strict_integer(row, "batch_size", row_number, "attention")
            prefill_chunk = _strict_integer(row, "prefill_chunk_size", row_number, "attention")
            cache_size = _strict_integer(row, "kv_cache_size", row_number, "attention")
            phase = "prefill" if prefill else "decode"
            query_tokens = prefill_chunk if prefill else 1
            context_tokens = prefill_chunk if prefill else cache_size + 1
            selector = FrozenDict(
                {
                    "semantic_operation": "attention_core",
                    "phase": phase,
                    "model_name": model_name,
                    "model_sequence_length": _strict_integer(row, "max_model_len", row_number, "attention"),
                    "hidden_size": _strict_integer(row, "n_embd", row_number, "attention"),
                    "attention_heads": _strict_integer(row, "n_q_head", row_number, "attention"),
                    "kv_heads": _strict_integer(row, "n_kv_head", row_number, "attention"),
                    "batch_size": batch_size,
                    "query_tokens": query_tokens,
                    "context_tokens": context_tokens,
                    "tensor_parallel": _strict_integer(row, "num_tensor_parallel_workers", row_number, "attention"),
                    "block_size": _strict_integer(row, "block_size", row_number, "attention"),
                    "implementation": row["attention_backend"],
                }
            )
            metrics = (
                (
                    "attention_core",
                    "time_stats.attn_prefill.median" if prefill else "time_stats.attn_decode.median",
                    "attention.core",
                ),
                ("attention_kv_cache_save", "time_stats.attn_kv_cache_save.median", "attention.kv_cache"),
            )
            for primitive, metric, source_layer in metrics:
                milliseconds = _milliseconds(row, metric)
                if milliseconds is None:
                    continue
                record_selector = selector.to_dict()
                record_selector["semantic_operation"] = primitive
                record_selector["source_layer"] = source_layer
                records.append(
                    PerformanceRecord(
                        record_id=cls._record_id(data_digest, "attention", row_number, metric),
                        subject=CostSubject.OPERATOR,
                        operation=primitive,
                        hardware=hardware_name,
                        datatype=datatype,
                        seconds=milliseconds * 1e-3,
                        selector=FrozenDict(record_selector),
                        provenance=provenance,
                        metadata=FrozenDict({"upstream_metric": metric}),
                    )
                )
        return tuple(records)

    @classmethod
    def _compute_records(
        cls,
        rows: tuple[dict[str, str], ...],
        *,
        model_name: str,
        hardware_name: str,
        datatype: str,
        provenance: EvidenceProvenance,
        data_digest: str,
    ) -> tuple[PerformanceRecord, ...]:
        records = []
        for row_number, row in enumerate(rows, start=1):
            shared_selector = {
                "model_name": model_name,
                "hidden_size": _strict_integer(row, "n_embd", row_number, "compute"),
                "feedforward_size": _strict_integer(row, "n_expanded_embd", row_number, "compute"),
                "attention_heads": _strict_integer(row, "n_head", row_number, "compute"),
                "kv_heads": _strict_integer(row, "n_kv_head", row_number, "compute"),
                "tensor_parallel": _strict_integer(row, "num_tensor_parallel_workers", row_number, "compute"),
                "num_tokens": _strict_integer(row, "num_tokens", row_number, "compute"),
                "use_gated_mlp": _strict_boolean(row, "use_gated_mlp", row_number, "compute"),
            }
            for primitive, metric in _COMPUTE_COLUMNS.items():
                milliseconds = _milliseconds(row, metric)
                if milliseconds is None:
                    continue
                operation = "gemm" if primitive in _GEMM_PRIMITIVES else primitive
                selector = {
                    **shared_selector,
                    "semantic_operation": primitive,
                    "source_layer": _SOURCE_LAYERS[primitive],
                }
                records.append(
                    PerformanceRecord(
                        record_id=cls._record_id(data_digest, "compute", row_number, metric),
                        subject=CostSubject.OPERATOR,
                        operation=operation,
                        hardware=hardware_name,
                        datatype=datatype,
                        seconds=milliseconds * 1e-3,
                        selector=FrozenDict(selector),
                        provenance=provenance,
                        metadata=FrozenDict({"upstream_metric": metric}),
                    )
                )
        return tuple(records)
