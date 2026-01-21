"""Model FLOPs calculations for blueprinting."""

import hyperparameter as hp

__all__ = ["ModelFlops"]


class ModelFlops:
    """Model FLOPs calculations."""

    def __init__(self, model: "Model"):
        self._model = model

    @hp.param("model")
    def mlp(self) -> int:
        """MLP FLOPs."""
        return 2 * self._model._params.mlp()

    @hp.param("model")
    def attn(self, use_qkv_bias=True, use_attn_bias=True) -> int:
        """Attention FLOPs."""
        m = self._model
        qkv_proj_input = m.hidden
        qkv_proj_output = m.attn_size

        attn_proj_input = m.attn_size * m.attn_heads
        attn_proj_output = m.hidden

        qkv_flops = 2 * 3 * m.attn_heads * qkv_proj_input * qkv_proj_output
        proj_flops = 2 * attn_proj_input * attn_proj_output
        mask_flops = 2 * m.seq_size * m.attn_heads * m.attn_size

        return qkv_flops + proj_flops + mask_flops

    @hp.param("model")
    def norm(self, type_norm="LN") -> int:
        """Normalization layer FLOPs."""
        m = self._model
        if type_norm == "LN":
            return 2 * m.hidden
        if type_norm == "RMS":
            return m.hidden
        return 0

    @hp.param("model")
    def embedding(self) -> int:
        """Embedding FLOPs."""
        return 0

    @hp.param("model")
    def total(self) -> int:
        """Total FLOPs."""
        return 0
