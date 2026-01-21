class Execution:
    def __init__(self, cfg) -> None:
        self.num_procs = cfg.num_procs | 0
        self.tensor_par = cfg.tensor_par | 0
        self.pipeline_par = cfg.pipeline_par | 0
        self.data_par = cfg.data_par | 0
        self.tensor_par_net = cfg.tensor_par_net | 0
        self.pipeline_par_net = cfg.pipeline_par_net | 0
        self.data_par_net = cfg.data_par_net | 0
        self.global_batch_size = cfg.batch_size | 0

        self.microbatch_size = cfg.microbatch_size | 0

        self.datatype = cfg.datatype | 0
        self.fused_activation = cfg.fused_activation | 0
        self.attention_type = cfg.attention_type | "multihead"  # ['multihead', 'multiquery']
        self.activation_recompute = cfg.activation_recompute | "none"  # ['full', 'attn_only', 'none']

        self.pipeline_interleaving = cfg.pipeline_interleaving | 0
        self.optimizer_sharding = cfg.optimizer_sharding | 0
        self.tensor_par_comm_type = cfg.tensor_par_comm_type | "ar"  # ['ar', 'p2p_rs_ag', 'rs_ag']

        self.tensor_par_overlap = cfg.tensor_par_overlap | "ring"  # ['none', 'ring', 'pipe']
        self.seq_par_ag_redo = cfg.seq_par_ag_redo | 0
        self.data_par_overlap = cfg.data_par_overlap | 0

        self.weight_offload = cfg.weight_offload | 0
        self.activations_offload = cfg.activations_offload | 0
        self.optimizer_offload = cfg.optimizer_offload | 0
        self.training = cfg.training | 0

        self._local_batch_size = self.global_batch_size // self.data_par
        self._sequence_par = self.tensor_par_comm_type == "rs_ag"
        self._pipeline_par_rs_ag = self.tensor_par_comm_type in ["p2p_rs_ag", "rs_ag"]
        self.in_network_reduction = False
        self._num_microbatches = self._local_batch_size // self.microbatch_size

    def get_json(self):
        return self.__dict__
