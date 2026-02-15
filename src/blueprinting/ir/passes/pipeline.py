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

参数管理:
- 使用 @hp.param("parallel") 从 hyperparameter scope 自动注入并行参数
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import hyperparameter as hp

from ...core.symbolic import sym_max
from ..types import OpNode, Phase, ScheduledOp, ScheduleIR
from .base import Pass


class PPScheduleMode(Enum):
    """PP 调度模式."""

    GPIPE = "gpipe"  # GPipe: F0 F1 F2 F3 B3 B2 B1 B0
    ONE_F_ONE_B = "1f1b"  # 1F1B: F0 F1 F2 F3 B3 F4 B2 F5 B1 ...
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

    @hp.param("parallel")
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        # P2P 通信参数
        p2p_bandwidth: float = 50e9,  # 50 GB/s (NVLink per direction)
        p2p_latency: float = 10e-6,  # 10µs 启动延迟
        dtype_bytes: int = 2,  # fp16
    ):
        """初始化 PipelineSchedulePass.

        Args:
            config: PP 调度配置
            p2p_bandwidth: P2P 通信带宽 (bytes/s)
            p2p_latency: P2P 通信延迟 (seconds)
            dtype_bytes: 数据类型字节数 (default: fp16 = 2)
        """
        self.config = config or PipelineConfig()
        self.p2p_bandwidth = p2p_bandwidth
        self.p2p_latency = p2p_latency
        self.dtype_bytes = dtype_bytes

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
            scheduled_ops = self._schedule_gpipe(
                forward_ops, backward_ops, num_mb, pp, ir
            )
        elif self.config.mode == PPScheduleMode.ONE_F_ONE_B:
            scheduled_ops = self._schedule_1f1b(
                forward_ops, backward_ops, num_mb, pp, ir
            )
        else:
            # 默认使用 1F1B
            scheduled_ops = self._schedule_1f1b(
                forward_ops, backward_ops, num_mb, pp, ir
            )

        # 创建新的 ScheduleIR
        new_ir = ScheduleIR(
            num_devices=ir.num_devices,
            metadata=ir.metadata.copy(),
            memory_pools=list(ir.memory_pools),  # 保留上游 Pass 生成的内存注解
        )
        new_ir.metadata["pp_mode"] = self.config.mode.value
        new_ir.metadata["num_microbatches"] = num_mb

        # 添加调度后的 Op
        for op in scheduled_ops:
            new_ir.add_op(op, stage=op.stage, device=op.device)

        # 添加优化器等其他 Op（在最后）
        # 每个 stage 的 optimizer 在该 stage 的 backward 完成后执行
        if scheduled_ops:
            final_time = 0
            for op in scheduled_ops:
                final_time = sym_max(final_time, op.start + op.duration)
        else:
            final_time = 0

        # 按 stage 分组 other_ops，保留原有的 stage 分配
        for op in other_ops:
            op.start = final_time
            # 保留 op 原有的 stage 和 device
            new_ir.add_op(op, stage=op.stage, device=op.device)

        # 计算 bubble time
        bubble_time = self._compute_bubble_time(scheduled_ops, num_mb, pp)
        new_ir.metadata["bubble_time"] = bubble_time

        return new_ir

    def _split_ops(
        self, ir: ScheduleIR
    ) -> Tuple[List[ScheduledOp], List[ScheduledOp], List[ScheduledOp]]:
        """将 Op 分为前向、反向、其他.

        使用 Phase 属性而非 op_type 后缀来分类：
        - Phase.FORWARD -> forward_ops
        - Phase.BACKWARD -> backward_ops (包含 recompute _RE 和 backward _BW)
        - Phase.OPTIMIZER -> other_ops
        """
        forward_ops = []
        backward_ops = []
        other_ops = []

        for op in ir.iter_ops():
            # 优先使用 Phase 属性
            if op.phase == Phase.FORWARD:
                forward_ops.append(op)
            elif op.phase == Phase.BACKWARD:
                # recompute (_RE) 和 backward (_BW) 都属于 BACKWARD phase
                backward_ops.append(op)
            elif op.phase == Phase.OPTIMIZER:
                other_ops.append(op)
            else:
                # 兼容旧代码：如果没有 Phase 属性，使用 op_type 判断
                op_type = op.op_type
                if "_BW" in op_type or "_RE" in op_type:
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
        ir: ScheduleIR,
    ) -> List[ScheduledOp]:
        """GPipe 调度: 所有 forward 完成后再做 backward.

        Schedule pattern (pp=4, mb=4):
        Device 0: F0 F1 F2 F3 ---- B3 B2 B1 B0
        Device 1: -- F0 F1 F2 F3 -- B3 B2 B1 B0
        Device 2: -- -- F0 F1 F2 F3 B3 B2 B1 B0
        Device 3: -- -- -- F0 F1 F2 F3 B3 B2 B1 B0
        """
        from ..types import Phase

        scheduled_ops = []

        # 按 stage 分组
        fw_by_stage = self._group_by_stage(forward_ops, pp)
        bw_by_stage = self._group_by_stage(backward_ops, pp)

        # 计算 P2P 通信时间（基于张量大小）
        batch_size = ir.metadata.get("batch_size", 1)
        seq_len = ir.metadata.get("seq_len", 2048)
        hidden = ir.metadata.get("hidden", 4096)
        activation_size = batch_size * seq_len * hidden
        activation_bytes = activation_size * self.dtype_bytes
        p2p_time = (
            self.p2p_latency + activation_bytes / self.p2p_bandwidth
            if self.p2p_bandwidth > 0
            else 0
        )

        # 每个 stage 的时间追踪
        stage_time = [0.0] * pp

        # Forward: mb0, mb1, mb2, ...
        for mb in range(num_mb):
            for stage in range(pp):
                # 等待前一个 stage 完成 + P2P 通信时间
                if stage > 0:
                    stage_time[stage] = sym_max(
                        stage_time[stage], stage_time[stage - 1] + 2 * p2p_time
                    )

                # 复制并调度该 stage 的前向 Op
                for op in fw_by_stage.get(stage, []):
                    new_op = self._copy_op_for_mb(op, mb, stage_time[stage])
                    stage_time[stage] = new_op.start + new_op.duration
                    scheduled_ops.append(new_op)

                # 添加 P2P Send/Recv（如果不是最后一个 stage）
                if stage < pp - 1:
                    send_op = self._create_p2p_op(
                        name=f"p2p_send_act_mb{mb}_s{stage}",
                        op_type="Send",
                        stage=stage,
                        phase=Phase.FORWARD,
                        start=stage_time[stage],
                        duration=p2p_time,
                        metadata={
                            "from_stage": stage,
                            "to_stage": stage + 1,
                            "mb": mb,
                            "data": "activation",
                            "data_size": activation_size,
                        },
                    )
                    scheduled_ops.append(send_op)

        # Backward: mb_{num_mb-1}, ..., mb1, mb0
        for mb in reversed(range(num_mb)):
            for stage in reversed(range(pp)):
                # 等待后一个 stage 完成 + P2P 通信时间
                if stage < pp - 1:
                    stage_time[stage] = sym_max(
                        stage_time[stage], stage_time[stage + 1] + 2 * p2p_time
                    )

                # 复制并调度该 stage 的反向 Op
                for op in bw_by_stage.get(stage, []):
                    new_op = self._copy_op_for_mb(op, mb, stage_time[stage])
                    stage_time[stage] = new_op.start + new_op.duration
                    scheduled_ops.append(new_op)

                # 添加 P2P Send/Recv（如果不是第一个 stage）
                if stage > 0:
                    send_op = self._create_p2p_op(
                        name=f"p2p_send_grad_mb{mb}_s{stage}",
                        op_type="Send",
                        stage=stage,
                        phase=Phase.BACKWARD,
                        start=stage_time[stage],
                        duration=p2p_time,
                        metadata={
                            "from_stage": stage,
                            "to_stage": stage - 1,
                            "mb": mb,
                            "data": "gradient",
                            "data_size": activation_size,
                        },
                    )
                    scheduled_ops.append(send_op)

        return scheduled_ops

    def _schedule_1f1b(
        self,
        forward_ops: List[ScheduledOp],
        backward_ops: List[ScheduledOp],
        num_mb: int,
        pp: int,
        ir: ScheduleIR,
    ) -> List[ScheduledOp]:
        """1F1B 调度: Pipeline Parallel 并行调度.

        1F1B (One Forward One Backward) 调度模式：

        每个 stage 的执行顺序：
        1. Warmup: 执行 (pp - 1 - stage) 个 Forward（填充 pipeline）
        2. Steady State: 交替执行 1B + 1F（稳态）
        3. Cooldown: 执行剩余的 Backward（排空 pipeline）

        关键约束：同一个 stage 在任意时刻只能执行一个操作。

        时间模型：
        - 每个 stage 有独立的本地时间线
        - stage 之间通过 P2P 通信同步
        - Stage i 的 Forward 需要等待 Stage i-1 的激活
        - Stage i 的 Backward 需要等待 Stage i+1 的梯度
        """
        from ..types import Phase

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

        # 从 metadata 获取激活张量大小（用于 P2P 通信）
        # 激活大小 = batch_size * seq_len * hidden * dtype_bytes
        batch_size = ir.metadata.get("batch_size", 1)
        seq_len = ir.metadata.get("seq_len", 2048)
        hidden = ir.metadata.get("hidden", 4096)
        activation_size = batch_size * seq_len * hidden  # 元素数
        activation_bytes = activation_size * self.dtype_bytes

        # 计算 P2P 通信时间（基于实际张量大小和带宽）
        # duration = latency + bytes / bandwidth
        p2p_time_act = (
            self.p2p_latency + activation_bytes / self.p2p_bandwidth
            if self.p2p_bandwidth > 0
            else 0
        )
        # 梯度与激活大小相同
        p2p_time_grad = p2p_time_act

        # 每个 stage 的当前时间（本地时间线）
        stage_current_time = [0.0] * pp

        # 记录每个 (stage, mb) 的 Forward 完成时间，用于 P2P 和 Backward 依赖
        fw_complete_time: Dict[Tuple[int, int], float] = {}
        # 记录每个 (stage, mb) 的 Backward 完成时间
        bw_complete_time: Dict[Tuple[int, int], float] = {}

        # 每个 stage 的下一个要执行的 Forward 和 Backward micro-batch
        next_fw_mb = [0] * pp
        next_bw_mb = [0] * pp

        def schedule_forward(stage: int, mb: int):
            """调度一个 Forward pass."""
            nonlocal stage_current_time

            # 计算开始时间：取决于
            # 1. 当前 stage 的本地时间（串行约束）
            # 2. 上一个 stage 发送的激活到达时间（数据依赖）
            start_time = stage_current_time[stage]

            if stage > 0:
                # 等待上一个 stage 的 Forward 完成 + P2P 通信时间 (Send + Recv)
                # 这样 Forward 计算在 P2P Recv 完成后才开始
                prev_fw_complete = fw_complete_time.get((stage - 1, mb), 0)
                start_time = sym_max(start_time, prev_fw_complete + 2 * p2p_time_act)

            # 调度 Forward ops
            current = start_time
            for op in fw_by_stage.get(stage, []):
                new_op = self._copy_op_for_mb(op, mb, current)
                current = new_op.start + new_op.duration
                scheduled_ops.append(new_op)

            fw_complete_time[(stage, mb)] = current
            stage_current_time[stage] = current

            # 添加 P2P Send/Recv（如果不是最后一个 stage）
            # 注意：P2P 通信与计算并行（不阻塞当前 stage 的下一个操作）
            if stage < pp - 1:
                send_op = self._create_p2p_op(
                    name=f"p2p_send_act_mb{mb}_s{stage}",
                    op_type="Send",
                    stage=stage,
                    phase=Phase.FORWARD,
                    start=current,
                    duration=p2p_time_act,
                    metadata={
                        "from_stage": stage,
                        "to_stage": stage + 1,
                        "mb": mb,
                        "data": "activation",
                        "data_size": activation_size,
                    },
                )
                scheduled_ops.append(send_op)
                # 不更新 stage_current_time，P2P 不阻塞计算

                recv_op = self._create_p2p_op(
                    name=f"p2p_recv_act_mb{mb}_s{stage+1}",
                    op_type="Recv",
                    stage=stage + 1,
                    phase=Phase.FORWARD,
                    start=current + p2p_time_act,
                    duration=p2p_time_act,
                    metadata={
                        "from_stage": stage,
                        "to_stage": stage + 1,
                        "mb": mb,
                        "data": "activation",
                        "data_size": activation_size,
                    },
                )
                scheduled_ops.append(recv_op)

        def schedule_backward(stage: int, mb: int):
            """调度一个 Backward pass."""
            nonlocal stage_current_time

            # 计算开始时间：取决于
            # 1. 当前 stage 的本地时间（串行约束）
            # 2. 当前 stage 的 Forward 完成时间（数据依赖）
            # 3. 下一个 stage 发送的梯度到达时间（数据依赖）
            start_time = stage_current_time[stage]
            start_time = sym_max(start_time, fw_complete_time.get((stage, mb), 0))

            if stage < pp - 1:
                # 等待下一个 stage 的 Backward 完成 + P2P 通信时间 (Send + Recv)
                # 这样 Backward 计算在 P2P Recv 完成后才开始
                next_bw_complete = bw_complete_time.get((stage + 1, mb), 0)
                start_time = sym_max(start_time, next_bw_complete + 2 * p2p_time_grad)

            # 调度 Backward ops
            current = start_time
            for op in bw_by_stage.get(stage, []):
                new_op = self._copy_op_for_mb(op, mb, current)
                current = new_op.start + new_op.duration
                scheduled_ops.append(new_op)

            bw_complete_time[(stage, mb)] = current
            stage_current_time[stage] = current

            # 添加 P2P Send/Recv（如果不是第一个 stage）
            # 注意：P2P 通信与计算并行（不阻塞当前 stage 的下一个操作）
            if stage > 0:
                send_op = self._create_p2p_op(
                    name=f"p2p_send_grad_mb{mb}_s{stage}",
                    op_type="Send",
                    stage=stage,
                    phase=Phase.BACKWARD,
                    start=current,
                    duration=p2p_time_grad,
                    metadata={
                        "from_stage": stage,
                        "to_stage": stage - 1,
                        "mb": mb,
                        "data": "gradient",
                        "data_size": activation_size,
                    },
                )
                scheduled_ops.append(send_op)
                # 不更新 stage_current_time，P2P 不阻塞计算

                recv_op = self._create_p2p_op(
                    name=f"p2p_recv_grad_mb{mb}_s{stage-1}",
                    op_type="Recv",
                    stage=stage - 1,
                    phase=Phase.BACKWARD,
                    start=current + p2p_time_grad,
                    duration=p2p_time_grad,
                    metadata={
                        "from_stage": stage,
                        "to_stage": stage - 1,
                        "mb": mb,
                        "data": "gradient",
                        "data_size": activation_size,
                    },
                )
                scheduled_ops.append(recv_op)

        # 1F1B 调度主循环
        #
        # 关键：调度顺序必须遵循数据依赖：
        # - Forward: stage 0 -> stage 1 -> ... -> stage pp-1 (激活传递方向)
        # - Backward: stage pp-1 -> ... -> stage 1 -> stage 0 (梯度传递方向)
        #
        # 每轮迭代中，每个 stage 尽可能多地调度操作（直到约束不满足）

        max_iterations = (num_mb + pp) * 2  # 防止无限循环
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            made_progress = False

            # 阶段 1: 调度 Forward (从低 stage 到高 stage)
            # 每个 stage 尽可能多地调度 Forward（直到约束不满足）
            for stage in range(pp):
                warmup_count = pp - 1 - stage

                # 循环调度该 stage 的所有可调度 Forward
                while next_fw_mb[stage] < num_mb:
                    mb = next_fw_mb[stage]

                    # 检查 1F1B 约束：in_flight 不能超过 warmup_count
                    in_flight = next_fw_mb[stage] - next_bw_mb[stage]
                    if in_flight > warmup_count:
                        break  # 需要先执行 Backward

                    # 检查数据依赖：上一个 stage 的 Forward 必须完成
                    if stage > 0 and (stage - 1, mb) not in fw_complete_time:
                        break  # 等待上一个 stage

                    schedule_forward(stage, mb)
                    next_fw_mb[stage] += 1
                    made_progress = True

            # 阶段 2: 调度 Backward (从高 stage 到低 stage)
            # 每个 stage 尽可能多地调度 Backward
            for stage in reversed(range(pp)):
                # 循环调度该 stage 的所有可调度 Backward
                while next_bw_mb[stage] < num_mb:
                    mb = next_bw_mb[stage]

                    # 检查数据依赖：
                    # 1. 当前 stage 的 Forward 必须完成
                    if (stage, mb) not in fw_complete_time:
                        break

                    # 2. 下一个 stage 的 Backward 必须完成（如果存在）
                    if stage < pp - 1 and (stage + 1, mb) not in bw_complete_time:
                        break

                    schedule_backward(stage, mb)
                    next_bw_mb[stage] += 1
                    made_progress = True

            # 检查是否完成
            all_done = all(
                next_fw_mb[s] >= num_mb and next_bw_mb[s] >= num_mb for s in range(pp)
            )
            if all_done:
                break

            if not made_progress:
                break

        return scheduled_ops

    def _create_p2p_op(
        self,
        name: str,
        op_type: str,
        stage: int,
        phase: "Phase",
        start: float,
        duration: float,
        metadata: Dict[str, Any],
    ) -> ScheduledOp:
        """创建 P2P 通信 Op (Send/Recv)."""
        from ..types import OpNode

        op_node = OpNode(
            name=name,
            op_type=op_type,
            attrs=metadata,
        )

        return ScheduledOp(
            op=op_node,
            device=stage,  # P2P Op 在对应 stage 的 device 上
            stage=stage,
            stream="comm",  # 通信流
            phase=phase,
            start=start,
            duration=duration,
        )

    def _group_by_stage(
        self, ops: List[ScheduledOp], pp: int
    ) -> Dict[int, List[ScheduledOp]]:
        """将 Op 按 stage 分组."""
        result: Dict[int, List[ScheduledOp]] = {}

        for op in ops:
            stage = op.stage
            if stage not in result:
                result[stage] = []
            result[stage].append(op)

        return result

    def _copy_op_for_mb(
        self, op: ScheduledOp, mb: int, start_time: float
    ) -> ScheduledOp:
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
            device=op.stage,  # PP 并行: device = stage，每个 stage 在独立 GPU 上
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
        total_time = 0
        for op in scheduled_ops:
            total_time = sym_max(total_time, op.start + op.duration)

        # 有效计算时间 (假设完美调度下的最小时间)
        total_compute = sum(op.duration for op in scheduled_ops)
        ideal_time = total_compute / pp  # 完美并行

        # Bubble time
        bubble = sym_max(0.0, total_time - ideal_time)

        return bubble
