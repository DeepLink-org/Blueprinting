"""Model module for blueprinting."""

import hyperparameter as hp

from .comm import ModelComm
from .flops import ModelFlops
from .params import ModelParams

__all__ = ["Model", "ModelParams", "ModelFlops", "ModelComm"]


class Model:
    """LLM Model configuration.

    Uses composition pattern instead of dynamic inheritance.

    Examples
    --------
    >>> import hyperparameter as hp
    >>> with hp.scope(hidden=512, feedforward=2048, attn_heads=8, attn_size=64, seq_size=1024, num_blocks=4):
    ...     m = Model()
    ...     m.hidden
    512
    >>> m.nparam_total
    39348224
    """

    @staticmethod
    def from_cfg(cfg=None) -> "Model":
        """Create Model from configuration.

        Args:
            cfg: hp.scope object or None to use current scope

        Returns:
            Model instance
        """
        return Model(cfg)

    def __init__(self, cfg=None) -> None:
        # Read configuration from hp.scope at runtime
        if cfg is None:
            cfg = hp.scope()
        self.hidden = cfg.hidden | 0
        self.feedforward = cfg.feedforward | 0
        self.seq_size = cfg.seq_size | 0
        self.attn_heads = cfg.attn_heads | 0
        self.attn_size = cfg.attn_size | 0
        self.num_blocks = cfg.num_blocks | 0

        # Composition pattern - delegate to specialized classes
        self._params = ModelParams(self)
        self._flops = ModelFlops(self)
        self._comm = ModelComm(self)

    def num_parameters(self) -> int:
        """Calculate total number of parameters.

        https://cs.stanford.edu/~matei/papers/2021/sc_megatron_lm.pdf
        Equation 2
        """
        p = 2 * self.hidden * self.feedforward  # MLP weights
        p += 4 * self.hidden * self.attn_heads * self.attn_size  # Attn weights
        p += self.hidden + self.feedforward  # biases MLP
        p += 3 * self.attn_heads * self.attn_size + self.hidden  # biases Attn
        p += 2 * 2 * self.hidden  # layer norm
        p *= self.num_blocks  # per each block
        p += (51200 + self.seq_size) * self.hidden  # embeddings
        return p

    # Delegate parameter calculations to ModelParams
    @property
    def nparam_mlp(self) -> int:
        """MLP parameters."""
        return self._params.mlp()

    @property
    def nparam_attn(self) -> int:
        """Attention parameters."""
        return self._params.attn()

    @property
    def nparam_norm(self) -> int:
        """Normalization layer parameters."""
        return self._params.norm()

    @property
    def nparam_embedding(self) -> int:
        """Embedding parameters."""
        return self._params.embedding()

    @property
    def nparam_total(self) -> int:
        """Total number of parameters in the model."""
        return self._params.total()

    # Delegate FLOPs calculations to ModelFlops
    @property
    def flops_mlp(self) -> int:
        """MLP FLOPs."""
        return self._flops.mlp()

    @property
    def flops_attn(self) -> int:
        """Attention FLOPs."""
        return self._flops.attn()

    @property
    def flops_norm(self) -> int:
        """Normalization layer FLOPs."""
        return self._flops.norm()

    @property
    def flops_embedding(self) -> int:
        """Embedding FLOPs."""
        return self._flops.embedding()

    @property
    def flops_total(self) -> int:
        """Total FLOPs."""
        return self._flops.total()

    # Delegate communication calculations to ModelComm
    @property
    def comm_embedding_fw(self):
        """Forward embedding communication."""
        return self._comm.embedding_fw()

    @property
    def comm_embedding_bw(self):
        """Backward embedding communication."""
        return self._comm.embedding_bw()

    @property
    def comm_attn_fw(self):
        """Forward attention communication."""
        return self._comm.attn_fw()

    @property
    def comm_attn_bw(self):
        """Backward attention communication."""
        return self._comm.attn_bw()

    @property
    def comm_mlp_fw(self):
        """Forward MLP communication."""
        return self._comm.mlp_fw()

    @property
    def comm_mlp_bw(self):
        """Backward MLP communication."""
        return self._comm.mlp_bw()

    @property
    def comm_total_fw(self):
        """Total forward communication."""
        return self._comm.total_fw()

    @property
    def comm_total_bw(self):
        """Total backward communication."""
        return self._comm.total_bw()

    @property
    def comm_pipeline_parallel_fw(self):
        """Forward pipeline parallel communication."""
        return self._comm.pipeline_parallel_fw()

    @property
    def comm_pipeline_parallel_bw(self):
        """Backward pipeline parallel communication."""
        return self._comm.pipeline_parallel_bw()
