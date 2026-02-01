"""OverlapAnalysisPass - Analyze compute-communication overlap.

这个 Pass 分析计算和通信的重叠:
1. 提取计算流和通信流的时间区间
2. 计算重叠时间
3. 评估重叠效率

对于有效的 compute-comm overlap:
- 需要硬件支持 (多流并发)
- 通信和计算可以同时执行
- 有效时间 = max(compute, comm) 而不是 compute + comm
"""

from typing import Dict, List, Tuple

from .base import Pass
from ..types import TimelineIR, EventType, StreamType


class OverlapAnalysisPass(Pass):
    """分析计算-通信重叠.
    
    这个 Pass 不修改 TimelineIR，只是分析并将结果存入 metadata:
    - compute_time: 总计算时间
    - comm_time: 总通信时间
    - overlap_time: 重叠时间
    - effective_time: 有效时间 (考虑重叠)
    - overlap_ratio: 重叠率
    """
    
    def __init__(self, overlap_efficiency: float = 0.9):
        """初始化 OverlapAnalysisPass.
        
        Args:
            overlap_efficiency: 重叠效率 (0.9 = 90% 的重叠可以有效利用)
        """
        self.overlap_efficiency = overlap_efficiency
    
    def run(self, ir: TimelineIR) -> TimelineIR:
        """执行重叠分析."""
        # 收集所有设备
        devices = set()
        for event in ir.events:
            devices.add(event.device)
        
        # 每个设备分析
        overlap_analysis = {}
        total_compute = 0.0
        total_comm = 0.0
        total_overlap = 0.0
        
        for device in devices:
            # 提取计算区间
            compute_intervals = self._get_intervals(ir, device, StreamType.COMPUTE)
            
            # 提取通信区间
            comm_intervals = self._get_intervals(ir, device, StreamType.COMM)
            
            # 计算重叠
            overlap = self._calculate_overlap(compute_intervals, comm_intervals)
            
            compute_time = sum(end - start for start, end in compute_intervals)
            comm_time = sum(end - start for start, end in comm_intervals)
            
            overlap_analysis[device] = {
                "compute_time": compute_time,
                "comm_time": comm_time,
                "overlap_time": overlap,
                "effective_time": compute_time + comm_time - overlap * self.overlap_efficiency,
                "overlap_ratio": overlap / comm_time if comm_time > 0 else 0,
            }
            
            total_compute += compute_time
            total_comm += comm_time
            total_overlap += overlap
        
        # 存储分析结果
        ir.metadata["overlap_analysis"] = {
            "per_device": overlap_analysis,
            "total": {
                "compute_time": total_compute,
                "comm_time": total_comm,
                "overlap_time": total_overlap,
                "overlap_efficiency": self.overlap_efficiency,
                "effective_time": total_compute + total_comm - total_overlap * self.overlap_efficiency,
                "overlap_ratio": total_overlap / total_comm if total_comm > 0 else 0,
            },
        }
        
        return ir
    
    def _get_intervals(
        self,
        ir: TimelineIR,
        device: int,
        stream: StreamType,
    ) -> List[Tuple[float, float]]:
        """提取指定设备和流的时间区间."""
        intervals = []
        starts: Dict[str, float] = {}
        
        for event in ir.events:
            if event.device != device:
                continue
            if event.stream != stream:
                continue
            
            # 只处理数值时间
            if not isinstance(event.time, (int, float)):
                continue
            
            resource = event.resource_id
            
            if event.event_type in (EventType.COMPUTE_START, EventType.COMM_START):
                starts[resource] = event.time
            
            elif event.event_type in (EventType.COMPUTE_END, EventType.COMM_END):
                if resource in starts:
                    intervals.append((starts[resource], event.time))
                    del starts[resource]
        
        return sorted(intervals)
    
    def _calculate_overlap(
        self,
        intervals1: List[Tuple[float, float]],
        intervals2: List[Tuple[float, float]],
    ) -> float:
        """计算两组区间的重叠时间."""
        if not intervals1 or not intervals2:
            return 0.0
        
        overlap = 0.0
        
        for start1, end1 in intervals1:
            for start2, end2 in intervals2:
                # 检查是否重叠
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                
                if overlap_start < overlap_end:
                    overlap += overlap_end - overlap_start
        
        return overlap
