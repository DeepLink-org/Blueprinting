"""ParallelPass - Apply parallel strategies to GraphIR.

这个 Pass 在 Graph IR 层应用并行策略:
1. 标记 Block 的 TP shard 策略 (tp_col, tp_row)
2. 分配 PP stage 和 device
3. 更新 metadata (tp, pp, dp)

不修改 workload 计算（由 SchedulePass 负责）。

参数管理:
- 使用 @hp.param("parallel") 从 hyperparameter scope 自动注入并行参数
"""

import re
from typing import Optional

import hyperparameter as hp

from ..types import BlockNode, GraphIR
from .base import Pass


class ParallelPass(Pass):
    """在 Graph IR 层应用并行策略.

    职责:
    - 标记 Block 的 shard 策略 (tp_col, tp_row, seq_par)
    - 分配 PP stage
    - 更新 metadata

    不负责:
    - workload 调整（由 SchedulePass 根据 shard 策略处理）
    - 插入通信 Op（由 ExpandPass 根据 shard 策略处理）
    """

    @hp.param("parallel")
    def __init__(
        self,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        tp_comm_type: str = "ar",  # "ar" 或 "rs_ag"
        sequence_parallel: bool = False,
    ):
        """初始化 ParallelPass.

        Args:
            tp: Tensor Parallelism 度数
            pp: Pipeline Parallelism 度数
            dp: Data Parallelism 度数
            tp_comm_type: TP 通信类型 ("ar" = AllReduce, "rs_ag" = ReduceScatter + AllGather)
            sequence_parallel: 是否使用序列并行
        """
        self.tp = tp
        self.pp = pp
        self.dp = dp
        self.tp_comm_type = tp_comm_type
        self.sequence_parallel = sequence_parallel

    def run(self, ir: GraphIR) -> GraphIR:
        """执行并行策略标记."""
        # 更新 metadata
        ir.metadata["tp"] = self.tp
        ir.metadata["pp"] = self.pp
        ir.metadata["dp"] = self.dp
        ir.metadata["tp_comm_type"] = self.tp_comm_type
        ir.metadata["sequence_parallel"] = self.sequence_parallel

        if ir.root:
            # 应用 TP 策略
            if self.tp > 1:
                self._apply_tensor_parallel(ir.root)

            # 应用 PP 策略
            if self.pp > 1:
                self._apply_pipeline_parallel(ir)

        return ir

    def _apply_tensor_parallel(self, block: BlockNode) -> None:
        """递归应用 TP 策略到 Block 树."""
        # 根据 Block 类型标记 shard 策略
        self._mark_tp_shard(block)

        # 递归处理子 Block
        for child in block.children:
            self._apply_tensor_parallel(child)

    def _mark_tp_shard(self, block: BlockNode) -> None:
        """标记单个 Block 的 shard 策略."""
        block_type = block.block_type

        # 已有 shard 标记则跳过
        if block.attrs.get("shard"):
            return

        # Attention 内的 Linear
        if block_type == "Linear":
            name = block.name.lower()

            # QKV projection: column parallel
            if any(
                x in name
                for x in ["qkv", "q_proj", "k_proj", "v_proj", "query", "key", "value"]
            ):
                block.attrs["shard"] = "tp_col"

            # Output projection: row parallel
            elif any(x in name for x in ["out", "o_proj", "output", "dense"]):
                block.attrs["shard"] = "tp_row"

            # FFN: up projection column, down projection row
            elif any(x in name for x in ["up", "gate", "fc1", "w1", "w3"]):
                block.attrs["shard"] = "tp_col"
            elif any(x in name for x in ["down", "fc2", "w2"]):
                block.attrs["shard"] = "tp_row"

        # Norm 层可以使用 sequence parallel
        elif block_type in ("RMSNorm", "LayerNorm") and self.sequence_parallel:
            block.attrs["shard"] = "seq_par"

    def _apply_pipeline_parallel(self, ir: GraphIR) -> None:
        """应用 PP 策略，分配 stage."""
        if not ir.root:
            return

        # 统计层数
        num_layers = ir.metadata.get("num_layers")
        if num_layers is None:
            # 从 root 的直接子节点推断
            num_layers = self._count_layers(ir.root)

        if num_layers == 0:
            return

        # 检查层数是否能被 PP 整除
        if num_layers % self.pp != 0:
            raise ValueError(
                f"Pipeline parallelism requires num_layers to be divisible by pp. "
                f"Got num_layers={num_layers}, pp={self.pp}."
            )

        layers_per_stage = num_layers // self.pp

        # 分配 stage
        self._assign_stages(ir.root, layers_per_stage)

    def _count_layers(self, root: BlockNode) -> int:
        """统计 TransformerLayer 的数量."""
        count = 0
        for child in root.children:
            if child.block_type in ("TransformerLayer", "Layer"):
                count += 1
            elif child.block_type in ("Transformer", "GPT", "LLaMA"):
                # 递归统计
                count += self._count_layers(child)
        return count

    def _assign_stages(self, block: BlockNode, layers_per_stage: int) -> None:
        """递归分配 stage."""
        for child in block.children:
            if child.block_type in ("TransformerLayer", "Layer"):
                # 从名称提取层索引
                layer_idx = self._extract_layer_index(child.name)
                if layer_idx is not None:
                    stage = min(layer_idx // layers_per_stage, self.pp - 1)
                    child.attrs["stage"] = stage
                    child.attrs["device"] = stage  # 简单映射：stage = device

            # 递归
            self._assign_stages(child, layers_per_stage)

    def _extract_layer_index(self, name: str) -> Optional[int]:
        """从名称提取层索引."""
        # 尝试 layer0, layer1, ...
        match = re.search(r"layer(\d+)", name, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # 尝试 block0, block1, ...
        match = re.search(r"block(\d+)", name, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None
