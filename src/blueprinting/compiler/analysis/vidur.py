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

from ..bindings import InferencePhase
from ..codec import content_digest
from ..frozen import FrozenDict
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
                    "adapter": "blueprinting-vidur-baseline-v1",
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
