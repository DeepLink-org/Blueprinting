"""Model communication calculations for blueprinting."""

import hyperparameter as hp

from ..counters import CommCounter

__all__ = ["ModelComm"]


class ModelComm:
    """Model communication calculations."""

    def __init__(self, model: "Model"):
        self._model = model

    @hp.param("exe")
    def embedding_fw(self, global_batch_size=0, microbatch_size=0, data_par=0, tensor_par=0) -> CommCounter:
        """Forward embedding communication."""
        m = self._model
        cnt = CommCounter()
        if tensor_par > 0:
            cnt.n_all_reduce += 1
            cnt.all_reduce += microbatch_size * m.seq_size * m.hidden

            cnt.n_all_reduce += 1
            cnt.all_reduce += microbatch_size * m.seq_size

            cnt.n_all_reduce += 1
            cnt.all_reduce += tensor_par
        return cnt

    @hp.param("exe")
    def embedding_bw(self, global_batch_size=0, microbatch_size=0, data_par=0, tensor_par=0) -> CommCounter:
        """Backward embedding communication."""
        return CommCounter()

    @hp.param("exe")
    def attn_fw(
        self,
        global_batch_size=0,
        microbatch_size=0,
        data_par=0,
        tensor_par=0,
        sequence_par=True,
    ) -> CommCounter:
        """Forward attention communication."""
        m = self._model
        cnt = CommCounter()
        if tensor_par > 0 and sequence_par == False:
            cnt.n_all_reduce += 1
            cnt.all_reduce += microbatch_size * m.seq_size * m.hidden
        elif tensor_par > 0 and sequence_par == True:
            cnt.n_all_gather += 1
            cnt.all_gather += microbatch_size * m.seq_size * m.hidden

            cnt.n_reduce_scatter += 1
            cnt.reduce_scatter += microbatch_size * m.seq_size * m.hidden
        return cnt

    @hp.param("exe")
    def attn_bw(
        self,
        global_batch_size=0,
        microbatch_size=0,
        data_par=0,
        tensor_par=0,
        sequence_par=True,
    ) -> CommCounter:
        """Backward attention communication."""
        m = self._model
        cnt = CommCounter()
        if tensor_par > 0 and sequence_par == False:
            cnt.n_all_reduce += 1
            cnt.all_reduce += microbatch_size * m.seq_size * m.hidden
        elif tensor_par > 0 and sequence_par == True:
            cnt.n_all_gather += 1
            cnt.all_gather += microbatch_size * m.seq_size * m.hidden

            cnt.n_reduce_scatter += 1
            cnt.reduce_scatter += microbatch_size * m.seq_size * m.hidden
        return cnt

    @hp.param("exe")
    def mlp_fw(
        self,
        microbatch_size=0,
        zero_optimizer=0,
        data_par=0,
        tensor_par=0,
        sequence_par=True,
    ) -> CommCounter:
        """Forward MLP communication."""
        m = self._model
        cnt = CommCounter()
        if tensor_par > 0 and sequence_par == False:
            # forward: attn -> all_reduce -> mlp
            cnt.n_all_reduce += 1
            cnt.all_reduce += microbatch_size * m.seq_size * m.hidden
        elif tensor_par > 0 and sequence_par == True:
            cnt.n_all_gather += 1
            cnt.all_gather += microbatch_size * m.seq_size * m.hidden

            cnt.n_reduce_scatter += 1
            cnt.reduce_scatter += microbatch_size * m.seq_size * m.hidden

        if data_par > 0 and zero_optimizer == 3:
            cnt.n_all_gather += 1
            cnt.all_gather += m.hidden * m.hidden
        return cnt

    @hp.param("exe")
    def mlp_bw(
        self,
        microbatch_size=0,
        zero_optimizer=0,
        data_par=0,
        tensor_par=0,
        sequence_par=True,
    ) -> CommCounter:
        """Backward MLP communication."""
        m = self._model
        cnt = CommCounter()
        if tensor_par > 0 and sequence_par == False:
            # backward: attn <- all_reduce <- mlp
            cnt.n_all_reduce += 1
            cnt.all_reduce += microbatch_size * m.seq_size * m.hidden
        elif tensor_par > 0 and sequence_par == True:
            cnt.n_all_gather += 1
            cnt.all_gather += microbatch_size * m.seq_size * m.hidden

            cnt.n_reduce_scatter += 1
            cnt.reduce_scatter += microbatch_size * m.seq_size * m.hidden

        if data_par > 0 and zero_optimizer == 0:
            # 梯度累加，梯度下发
            cnt.n_all_reduce += 1
            cnt.all_reduce += m.hidden * m.hidden
        elif data_par > 0 and zero_optimizer == 1:
            # 梯度累加，梯度下发
            cnt.n_all_reduce += 1
            cnt.all_reduce += m.hidden * m.hidden

            # 参数聚合
            cnt.n_all_gather += 1
            cnt.all_gather += m.hidden * m.hidden
        elif data_par > 0 and zero_optimizer == 2:
            # 梯度聚合
            cnt.n_reduce_scatter += 1
            cnt.reduce_scatter += m.hidden * m.hidden

            # 参数聚合
            cnt.n_all_gather += 1
            cnt.all_gather += m.hidden * m.hidden
        elif data_par > 0 and zero_optimizer == 3:
            # 参数聚合
            cnt.n_all_gather += 1
            cnt.all_gather += m.hidden * m.hidden

            # 梯度聚合
            cnt.n_reduce_scatter += 1
            cnt.reduce_scatter += m.hidden * m.hidden
        return cnt

    @hp.param("exe")
    def pipeline_parallel_fw(self, microbatch_size=0, pipeline_par=0) -> CommCounter:
        """Forward pipeline parallel communication."""
        return CommCounter()

    @hp.param("exe")
    def pipeline_parallel_bw(self, microbatch_size=0, pipeline_par=0) -> CommCounter:
        """Backward pipeline parallel communication."""
        return CommCounter()

    def total_fw(self) -> CommCounter:
        """Total forward communication."""
        return self.embedding_fw() + self.attn_fw() + self.mlp_fw()

    def total_bw(self) -> CommCounter:
        """Total backward communication."""
        return self.embedding_bw() + self.attn_bw() + self.mlp_bw()
