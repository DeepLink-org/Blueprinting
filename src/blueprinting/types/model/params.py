"""Model parameter calculations for blueprinting."""

import hyperparameter as hp

__all__ = ["ModelParams"]


class ModelParams:
    """Model parameter calculations.

    Examples
    --------
    >>> from blueprinting.types.model import Model
    >>> import hyperparameter as hp
    >>> with hp.scope(hidden=512, feedforward=2048, attn_heads=8, attn_size=64, seq_size=1024, num_blocks=4):
    ...     m = Model()
    ...     m.nparam_mlp
    2099712
    """

    def __init__(self, model: "Model"):
        self._model = model

    @hp.param("model")
    def mlp(self, use_mlp_bias=True) -> int:
        """MLP parameters.

        Parameters
        ----------
        use_mlp_bias : bool, optional
            Whether to include bias parameters, by default True
        """
        m = self._model
        bias = m.hidden + m.feedforward if use_mlp_bias else 0
        return 2 * m.feedforward * m.hidden + bias

    @hp.param("model")
    def attn(self, use_qkv_bias=True, use_attn_bias=True) -> int:
        """Attention parameters.

        Parameters
        ----------
        use_qkv_bias : bool, optional
            Whether to include bias in QKV projection, by default True
        use_attn_bias : bool, optional
            Whether to include bias in attention projection, by default True
        """
        m = self._model
        qkv_proj_input = m.hidden
        qkv_proj_output = m.attn_size
        qkv_proj_bias = m.attn_size if use_qkv_bias else 0

        attn_proj_input = m.attn_size * m.attn_heads
        attn_proj_output = m.hidden
        attn_proj_bias = m.hidden if use_attn_bias else 0

        return 3 * m.attn_heads * (qkv_proj_input * qkv_proj_output + qkv_proj_bias) + (
            attn_proj_input * attn_proj_output + attn_proj_bias
        )

    @hp.param("model")
    def norm(self, type_norm="LN") -> int:
        """Normalization layer parameters."""
        m = self._model
        if type_norm == "LN":
            input_norm = 2 * m.hidden
            attn_norm = 2 * m.hidden
        elif type_norm == "RMS":
            input_norm = 0
            attn_norm = 0
        else:
            input_norm = 0
            attn_norm = 0
        return input_norm + attn_norm

    @hp.param("model")
    def embedding(self, vocab_size=51200, type_posemb="learned") -> int:
        """Embedding parameters."""
        m = self._model
        if type_posemb == "learned":
            posemb = m.seq_size * m.hidden
        elif type_posemb == "rope":
            posemb = 0
        else:
            posemb = 0
        return vocab_size * m.hidden + posemb

    def total(self) -> int:
        """Total number of parameters in the model."""
        m = self._model
        return m.num_blocks * (self.mlp() + self.attn() + self.norm()) + self.embedding()
