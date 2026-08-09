"""Adapters for NVIDIA AIConfigurator operator performance tables.

AIConfigurator uses one schema per operation family.  This adapter supports
the stable GEMM, context/decode attention, and custom all-reduce schemas
explicitly instead of guessing columns from an arbitrary Parquet file.
"""

from __future__ import annotations

import hashlib
import math
from enum import Enum
from pathlib import Path
from typing import Any

from ...compiler.codec import content_digest
from ...compiler.frozen import FrozenDict
from .database import EvidenceProvenance, PerformanceDatabase, PerformanceRecord
from .importers import read_tabular_rows
from .protocol import CostSubject, EstimateMethod


class AIConfiguratorTable(Enum):
    GEMM = "gemm"
    CONTEXT_ATTENTION = "context_attention"
    GENERATION_ATTENTION = "generation_attention"
    CUSTOM_ALL_REDUCE = "custom_allreduce"


_FILE_TABLES = {
    "gemm_perf.parquet": AIConfiguratorTable.GEMM,
    "gemm_perf.csv": AIConfiguratorTable.GEMM,
    "context_attention_perf.parquet": AIConfiguratorTable.CONTEXT_ATTENTION,
    "context_attention_perf.csv": AIConfiguratorTable.CONTEXT_ATTENTION,
    "generation_attention_perf.parquet": AIConfiguratorTable.GENERATION_ATTENTION,
    "generation_attention_perf.csv": AIConfiguratorTable.GENERATION_ATTENTION,
    "custom_allreduce_perf.parquet": AIConfiguratorTable.CUSTOM_ALL_REDUCE,
    "custom_allreduce_perf.csv": AIConfiguratorTable.CUSTOM_ALL_REDUCE,
}

_DATATYPE_ALIASES = {
    "half": "float16",
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


def _required(row: dict[str, Any], column: str, row_number: int) -> Any:
    if column not in row:
        raise ValueError(f"AIConfigurator row {row_number} is missing column {column!r}")
    value = row[column]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"AIConfigurator row {row_number} has an empty {column!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"AIConfigurator row {row_number} has non-finite {column!r}")
    return value


def _integer(row: dict[str, Any], column: str, row_number: int) -> int:
    value = _required(row, column, row_number)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"AIConfigurator row {row_number} has non-numeric {column!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"AIConfigurator row {row_number} has non-integer {column!r}")
    return int(numeric)


def _text(row: dict[str, Any], column: str, row_number: int) -> str:
    return str(_required(row, column, row_number)).strip()


def _datatype(row: dict[str, Any], column: str, row_number: int) -> str:
    value = _text(row, column, row_number).lower()
    return _DATATYPE_ALIASES.get(value, value)


def _latency_seconds(row: dict[str, Any], row_number: int) -> float:
    value = _required(row, "latency", row_number)
    try:
        latency_ms = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"AIConfigurator row {row_number} has invalid latency") from error
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError(f"AIConfigurator row {row_number} has invalid latency")
    return latency_ms * 1e-3


def _runtime_selector(row: dict[str, Any], row_number: int) -> dict[str, Any]:
    return {
        "runtime": _text(row, "framework", row_number).lower(),
        "runtime_revision": _text(row, "version", row_number),
        "implementation": _text(row, "kernel_source", row_number),
    }


class AIConfiguratorPerformanceImporter:
    IMPORTER_REVISION = "blueprinting-aiconfigurator-perf-v1"

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        hardware_name: str,
        source_revision: str,
        table: AIConfiguratorTable | None = None,
        database_name: str | None = None,
    ) -> PerformanceDatabase:
        source = Path(path)
        if not isinstance(hardware_name, str) or not hardware_name:
            raise ValueError("hardware_name must be a non-empty string")
        if not isinstance(source_revision, str) or not source_revision:
            raise ValueError("source_revision must be a non-empty string")
        if table is None:
            table = _FILE_TABLES.get(source.name)
            if table is None:
                raise ValueError("cannot infer AIConfigurator table family from file name; pass table explicitly")
        if not isinstance(table, AIConfiguratorTable):
            raise TypeError("table must be AIConfiguratorTable")
        payload = source.read_bytes()
        data_digest = hashlib.sha256(payload).hexdigest()
        rows = read_tabular_rows(source)
        provenance = EvidenceProvenance(
            source="nvidia-aiconfigurator",
            source_revision=source_revision,
            importer=cls.IMPORTER_REVISION,
            data_digest=data_digest,
            method=EstimateMethod.MEASURED,
            metadata=FrozenDict({"table": table.value, "file_name": source.name}),
        )
        records = tuple(
            cls._record(
                row,
                row_number,
                table=table,
                hardware_name=hardware_name,
                provenance=provenance,
                data_digest=data_digest,
            )
            for row_number, row in enumerate(rows, start=1)
        )
        return PerformanceDatabase(
            name=database_name or f"aiconfigurator-{hardware_name}-{table.value}",
            records=records,
            metadata=FrozenDict(
                {
                    "source": "nvidia-aiconfigurator",
                    "source_revision": source_revision,
                    "data_digest": data_digest,
                    "importer": cls.IMPORTER_REVISION,
                    "table": table.value,
                }
            ),
        )

    @classmethod
    def _record(
        cls,
        row: dict[str, Any],
        row_number: int,
        *,
        table: AIConfiguratorTable,
        hardware_name: str,
        provenance: EvidenceProvenance,
        data_digest: str,
    ) -> PerformanceRecord:
        selector = _runtime_selector(row, row_number)
        metadata: dict[str, Any] = {
            "device": _text(row, "device", row_number),
            "upstream_operation": _text(row, "op_name", row_number),
            "source_table": table.value,
        }
        if table is AIConfiguratorTable.GEMM:
            operation = "gemm"
            subject = CostSubject.OPERATOR
            datatype = _datatype(row, "gemm_dtype", row_number)
            selector.update(
                {
                    "m": _integer(row, "m", row_number),
                    "n": _integer(row, "n", row_number),
                    "k": _integer(row, "k", row_number),
                }
            )
        elif table in {AIConfiguratorTable.CONTEXT_ATTENTION, AIConfiguratorTable.GENERATION_ATTENTION}:
            operation = "attention_core"
            subject = CostSubject.OPERATOR
            datatype = _datatype(row, "attn_dtype", row_number)
            is_context = table is AIConfiguratorTable.CONTEXT_ATTENTION
            input_length = _integer(row, "isl", row_number)
            step = _integer(row, "step", row_number)
            selector.update(
                {
                    "semantic_operation": "attention_core",
                    "phase": "prefill" if is_context else "decode",
                    "batch_size": _integer(row, "batch_size", row_number),
                    "query_tokens": input_length if is_context else 1,
                    "context_tokens": input_length if is_context else input_length + step,
                    "local_attention_heads": _integer(row, "num_heads", row_number),
                    "local_kv_heads": _integer(row, "num_key_value_heads", row_number),
                    "head_size": _integer(row, "head_dim", row_number),
                    "beam_width": _integer(row, "beam_width", row_number),
                    "window_size": _integer(row, "window_size", row_number),
                    "kv_cache_datatype": _datatype(row, "kv_cache_dtype", row_number),
                }
            )
        else:
            operation = "all_reduce"
            subject = CostSubject.COMMUNICATION
            datatype = _datatype(row, "allreduce_dtype", row_number)
            selector.update(
                {
                    "participants": _integer(row, "num_gpus", row_number),
                    "message_bytes": _integer(row, "message_size", row_number),
                    "backend": _text(row, "backend", row_number),
                }
            )
        for optional in ("power", "power_limit"):
            if optional in row and row[optional] is not None:
                value = float(row[optional])
                if math.isfinite(value):
                    metadata[optional] = value
        record_id = content_digest(
            FrozenDict(
                {
                    "importer": cls.IMPORTER_REVISION,
                    "data_digest": data_digest,
                    "table": table.value,
                    "row": row_number,
                }
            ),
            "performance-record-id",
        )
        return PerformanceRecord(
            record_id=record_id,
            subject=subject,
            operation=operation,
            hardware=hardware_name,
            datatype=datatype,
            seconds=_latency_seconds(row, row_number),
            selector=FrozenDict(selector),
            provenance=provenance,
            metadata=FrozenDict(metadata),
        )
