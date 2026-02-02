"""PipelineSchedulePass - Pipeline Parallelism scheduling modes.

这个 Pass 负责 PP 场景下的调度:
1. 将单个 micro-batch 的 Op 序列复制为多个 micro-batch
2. 根据调度模式（GPipe, 1F1B 等）交错排列 Op
3. 插入 P2P 通信 Op（Send/Recv）
4. 计算 bubble time

PP 调度模式:
- GPipe: 所有 forward 完成后再做 backward
- 1F1B: 一个 forward 后紧跟一个 backward（稳态时）
- Interleaved 1F1B: 每个设备持有多个 stage
- Zero Bubble: 将 BW 拆分为 W 和 B，最小化 bubble
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from copy import deepcopy

from .base import Pass
from ..types import ScheduleIR, ScheduledOp, OpNode, Phase


class PPScheduleMode(Enum):
    """PP 调度模式."""
    GPIPE = "gpipe"           # GPipe: F0 F1 F2 F3 B3 B2 B1 B0
    ONE_F_ONE_B = "1f1b"      # 1F1B: F0 F1 F2 F3 B3 F4 B2 F5 B1 ...
    INTERLEAVED = "interleaved"  # Interleaved 1F1B: 每个设备多个 virtual stage
    ZERO_BUBBLE = "zero_bubble"  # Zero Bubble PP


@dataclass
class PipelineConfig:
    """PP 调度配置.
    
    Attributes:
        mode: 调度模式
        num_microbatches: micro-batch 数量
        num_stages: stage 数量 (= PP 度数)
        virtual_stages_per_device: 每个设备的虚拟 stage 数（Interleaved 模式）
    """
    mode: PPScheduleMode = PPScheduleMode.ONE_F_ONE_B
    num_microbatches: int = 1
    num_stages: int = 1
    virtual_stages_per_device: int = 1


class PipelineSchedulePass(Pass):
    """处理 PP 调度模式.
    
    工作流程:
    1. 接收单个 micro-batch 的 ScheduleIR
    2. 复制为 num_microbatches 份
    3. 根据 mode 交错排列
    4. 插入 P2P 通信
    5. 重新计算时间
    
    输入: ScheduleIR (单个 micro-batch 的前向 + 反向 Op)
    输出: ScheduleIR (多个 micro-batch 交错后的 Op)
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        # P2P 通信参数
        p2p_bandwidth: float = 50e9,  # 50 GB/s (NVLink per direction)
    ):
        """初始化 PipelineSchedulePass.
        
        Args:
            config: PP 调度配置
            p2p_bandwidth: P2P 通信带宽 (bytes/s)
        """
        self.config = config or PipelineConfig()
        self.p2p_bandwidth = p2p_bandwidth
    
    def run(self, ir: ScheduleIR) -> ScheduleIR:
        """执行 PP 调度."""
        pp = ir.metadata.get("pp", 1)
        num_mb = self.config.num_microbatches
        
        # PP=1 或 micro-batch=1 时跳过
        if pp <= 1 or num_mb <= 1:
            ir.metadata["pp_mode"] = "none"
            ir.metadata["num_microbatches"] = num_mb
            return ir
        
        # 更新配置
        self.config.num_stages = pp
        
        # 分离前向和反向 Op
        forward_ops, backward_ops, other_ops = self._split_ops(ir)
        
        # 根据模式生成调度
        if self.config.mode == PPScheduleMode.GPIPE:
            scheduled_ops = self._schedule_gpipe(forward_ops, backward_ops, num_mb, pp)
        elif self.config.mode == PPScheduleMode.ONE_F_ONE_B:
            scheduled_ops = self._schedule_1f1b(forward_ops, backward_ops, num_mb, pp)
        else:
            # 默认使用 1F1B
            scheduled_ops = self._schedule_1f1b(forward_ops, backward_ops, num_mb, pp)
        
        # 创建新的 ScheduleIR
        new_ir = ScheduleIR(
            num_devices=ir.num_devices,
            metadata=ir.metadata.copy(),
        )
        new_ir.metadata["pp_mode"] = self.config.mode.value
        new_ir.metadata["num_microbatches"] = num_mb
        
        # 添加调度后的 Op
        for op in scheduled_ops:
            new_ir.add_op(op, stage=op.stage, device=op.device)
        
        # 添加优化器等其他 Op（在最后）
        if scheduled_ops:
            final_time = max(op.start + op.duration for op in scheduled_ops)
        else:
            final_time = 0
        
        for op in other_ops:
            op.start = final_time
            final_time += op.duration
            new_ir.add_op(op, stage=0, device=0)
        
        # 计算 bubble time
        bubble_time = self._compute_bubble_time(scheduled_ops, num_mb, pp)
        new_ir.metadata["bubble_time"] = bubble_time
        
        
        return new_ir
    
    def _split_ops(self, ir: ScheduleIR) -> Tuple[List[ScheduledOp], List[ScheduledOp], List[ScheduledOp]]:
        """将 Op 分为前向、反向、其他."""
        forward_ops = []
        backward_ops = []
        other_ops = []
        
        for op in ir.iter_ops():
            op_type = op.op_type
            if "_BW" in op_type:
                backward_ops.append(op)
            elif op_type in ("OptimizerStep", "RecomputeStep"):
                other_ops.append(op)
            else:
                forward_ops.append(op)
        
        return forward_ops, backward_ops, other_ops
    
    def _schedule_gpipe(
        self,
        forward_ops: List[ScheduledOp],
        backward_ops: List[ScheduledOp],
        num_mb: int,
        pp: int,
    ) -> List[ScheduledOp]:
        """GPipe 调度: 所有 forward 完成后再做 backward.
        
        Schedule pattern (pp=4, mb=4):
        Device 0: F0 F1 F2 F3 ---- B3 B2 B1 B0
        Device 1: -- F0 F1 F2 F3 -- B3 B2 B1 B0
        Device 2: -- -- F0 F1 F2 F3 B3 B2 B1 B0
        Device 3: -- -- -- F0 F1 F2 F3 B3 B2 B1 B0
        """
        scheduled_ops = []
        
        # 按 stage 分组
        fw_by_stage = self._group_by_stage(forward_ops, pp)
        bw_by_stage = self._group_by_stage(backward_ops, pp)
        
        # 每个 stage 的时间追踪
        stage_time = [0.0] * pp
        
        # Forward: mb0, mb1, mb2, ...
        for mb in range(num_mb):
            for stage in range(pp):
                # 等待前一个 stage 完成
                if stage > 0:
                    stage_time[stage] = max(stage_time[stage], stage_time[stage - 1])
                
                # 复制并调度该 stage 的前向 Op
                for op in fw_by_stage.get(stage, []):
                    new_op = self._copy_op_for_mb(op, mb, stage_time[stage])
                    stage_time[stage] = new_op.start + new_op.duration
                    scheduled_ops.append(new_op)
        
        # Backward: mb_{num_mb-1}, ..., mb1, mb0
        for mb in reversed(range(num_mb)):
            for stage in reversed(range(pp)):
                # 等待后一个 stage 完成
                if stage < pp - 1:
                    stage_time[stage] = max(stage_time[stage], stage_time[stage + 1])
                
                # 复制并调度该 stage 的反向 Op
                for op in bw_by_stage.get(stage, []):
                    new_op = self._copy_op_for_mb(op, mb, stage_time[stage])
                    stage_time[stage] = new_op.start + new_op.duration
                    scheduled_ops.append(new_op)
        
        return scheduled_ops
    
    def _schedule_1f1b(
        self,
        forward_ops: List[ScheduledOp],
        backward_ops: List[ScheduledOp],
        num_mb: int,
        pp: int,
    ) -> List[ScheduledOp]:
        """1F1B 调度: Pipeline Parallel 并行调度.
        
        正确的 Pipeline 调度：
        - 不同 stage 并行处理不同的 micro-batch
        - Stage i 在时间 t 处理的 micro-batch = Stage 0 在时间 t-i*stage_time 处理的
        
        时间模型:
        - per_stage_time = max(fw_time, bw_time) for 该 stage
        - 总时间 = (num_mb + pp - 1) × per_stage_time
        - bubble_time = (pp - 1) × per_stage_time
        """
        scheduled_ops = []
        
        # 按 stage 分组
        fw_by_stage = self._group_by_stage(forward_ops, pp)
        bw_by_stage = self._group_by_stage(backward_ops, pp)
        
        # 计算每个 stage 的时间
        stage_fw_time = {}
        stage_bw_time = {}
        for stage in range(pp):
            stage_fw_time[stage] = sum(op.duration for op in fw_by_stage.get(stage, []))
            stage_bw_time[stage] = sum(op.duration for op in bw_by_stage.get(stage, []))
        
        # 使用最大 stage 时间作为基准（所有 stage 同步）
        max_fw_time = max(stage_fw_time.values()) if stage_fw_time else 0
        max_bw_time = max(stage_bw_time.values()) if stage_bw_time else 0
        per_stage_time = max_fw_time + max_bw_time
        
        
        # 为每个 stage 生成独立的时间线
        # Stage i 的起始偏移 = i × per_stage_time（pipeline filling）
        
        for stage in range(pp):
            # Stage offset（pipeline 延迟）
            stage_offset = stage * max_fw_time
            
            # 该 stage 的本地时间
            local_time = 0.0
            
            # Forward: 处理所有 micro-batch
            for mb in range(num_mb):
                # 开始时间 = stage_offset + local_time
                start_time = stage_offset + local_time
                
                for op in fw_by_stage.get(stage, []):
                    new_op = self._copy_op_for_mb(op, mb, start_time)
                    start_time = new_op.start + new_op.duration
                    scheduled_ops.append(new_op)
                
                # 更新本地时间（一个 stage 的 forward+backward 总时间）
                # 在稳态，一个 micro-batch 占用 max(fw, bw) 时间
                local_time += max_fw_time
            
            # Backward: 反向处理所有 micro-batch
            # Backward 开始时间 = 该 micro-batch 的 forward 结束 + (pp-1-stage) × bw_time
            local_time = 0.0
            for mb in range(num_mb):
                # Backward 开始时间考虑：
                # 1. 该 stage 的 forward 已完成
                # 2. 后面的 stage 的 backward 已完成
                fw_end = stage_offset + (mb + 1) * max_fw_time
                bw_delay = (pp - 1 - stage) * max_bw_time
                start_time = fw_end + bw_delay + local_time
                
                for op in bw_by_stage.get(stage, []):
                    new_op = self._copy_op_for_mb(op, mb, start_time)
                    start_time = new_op.start + new_op.duration
                    scheduled_ops.append(new_op)
                
                local_time += max_bw_time
        
        return scheduled_ops
    
    def _group_by_stage(self, ops: List[ScheduledOp], pp: int) -> Dict[int, List[ScheduledOp]]:
        """将 Op 按 stage 分组."""
        result: Dict[int, List[ScheduledOp]] = {}
        
        for op in ops:
            stage = op.stage
            if stage not in result:
                result[stage] = []
            result[stage].append(op)
        
        return result
    
    def _copy_op_for_mb(self, op: ScheduledOp, mb: int, start_time: float) -> ScheduledOp:
        """为特定 micro-batch 复制 Op."""
        new_name = f"{op.op.name}_mb{mb}" if op.op else f"op_mb{mb}"
        
        new_op_node = OpNode(
            name=new_name,
            op_type=op.op_type,
            inputs=op.op.inputs if op.op else [],
            outputs=op.op.outputs if op.op else [],
            attrs=op.op.attrs.copy() if op.op else {},
            source_block=op.op.source_block if op.op else None,
            flops=op.op.flops if op.op else None,
            memory_bytes=op.op.memory_bytes if op.op else None,
            comm_bytes=op.op.comm_bytes if op.op else None,
        )
        new_op_node.attrs["micro_batch"] = mb
        
        return ScheduledOp(
            op=new_op_node,
            device=op.device,
            stage=op.stage,
            stream=op.stream,
            phase=op.phase,  # 保留原 op 的 phase
            start=start_time,
            duration=op.duration,
        )
    
    def _compute_bubble_time(
        self,
        scheduled_ops: List[ScheduledOp],
        num_mb: int,
        pp: int,
    ) -> float:
        """计算 bubble time.
        
        Bubble time = 总时间 - 有效计算时间
        
        GPipe bubble ratio ≈ (pp - 1) / num_mb
        1F1B bubble ratio ≈ (pp - 1) / num_mb (but better in practice)
        """
        if not scheduled_ops:
            return 0.0
        
        # 总时间
        total_time = max(op.start + op.duration for op in scheduled_ops)
        
        # 有效计算时间 (假设完美调度下的最小时间)
        total_compute = sum(op.duration for op in scheduled_ops)
        ideal_time = total_compute / pp  # 完美并行
        
        # Bubble time
        bubble = max(0.0, total_time - ideal_time)
        
        return bubble
