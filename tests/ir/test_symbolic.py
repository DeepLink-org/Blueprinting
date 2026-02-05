"""Unit tests for symbolic computation compatibility.

测试新的三层 IR 架构是否与 SymPy 符号运算兼容。
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sympy import Symbol, symbols, Expr

from src.blueprinting.ir.types import (
    BlockNode, GraphIR,
    OpNode, ScheduledOp, ScheduleIR,
    TimelineEvent, TimelineIR, EventType, StreamType,
)
from src.blueprinting.ir.ops import Linear, RMSNorm, Matmul
from src.blueprinting.ir.dsl import Model, Transformer
from blueprinting.core import SymMax, clear_expr_cache


# ==============================================================================
# Symbolic BlockDef Tests
# ==============================================================================

class TestSymbolicBlockDef:
    """测试 BlockDef 的符号计算能力."""
    
    def test_linear_symbolic_flops(self):
        """测试 Linear 使用符号参数计算 FLOPs."""
        B, S, H = symbols('B S H', positive=True, integer=True)
        
        attrs = {
            "in_features": H,
            "out_features": 4 * H,
            "batch_seq": B * S,
            "bias": True,
        }
        
        flops = Linear.compute_flops(attrs)
        
        # flops = 2 * B*S * H * 4*H + B*S * 4*H
        #       = 8 * B*S*H^2 + 4*B*S*H
        assert isinstance(flops, Expr)
        
        # 代入具体值验证
        concrete = flops.subs({B: 4, S: 2048, H: 4096})
        expected = 2 * 4 * 2048 * 4096 * 4 * 4096 + 4 * 2048 * 4 * 4096
        assert concrete == expected
    
    def test_linear_symbolic_params(self):
        """测试 Linear 使用符号参数计算参数内存."""
        H = Symbol('H', positive=True, integer=True)
        
        attrs = {
            "in_features": H,
            "out_features": 4 * H,
            "bias": True,
        }
        
        param_bytes = Linear.compute_param_bytes(attrs)
        
        # param_bytes = H * 4*H * 2 + 4*H * 2 = 8*H^2 + 8*H
        assert isinstance(param_bytes, Expr)
        
        # 代入具体值
        concrete = param_bytes.subs({H: 4096})
        expected = 4096 * 4 * 4096 * 2 + 4 * 4096 * 2
        assert concrete == expected
    
    def test_rmsnorm_symbolic_flops(self):
        """测试 RMSNorm 使用符号参数."""
        B, S, H = symbols('B S H', positive=True, integer=True)
        
        attrs = {
            "normalized_shape": H,
            "batch_seq": B * S,
        }
        
        flops = RMSNorm.compute_flops(attrs)
        
        # flops = 5 * batch_seq * hidden
        assert isinstance(flops, Expr)
        
        concrete = flops.subs({B: 4, S: 2048, H: 4096})
        expected = 5 * 4 * 2048 * 4096
        assert concrete == expected


class TestSymbolicOpDef:
    """测试 OpDef 的符号计算能力."""
    
    def test_matmul_symbolic_flops(self):
        """测试 Matmul 使用符号参数."""
        M, K, N = symbols('M K N', positive=True, integer=True)
        
        attrs = {"M": M, "K": K, "N": N}
        
        flops = Matmul.compute_flops(attrs)
        
        # flops = 2 * M * K * N
        assert isinstance(flops, Expr)
        
        concrete = flops.subs({M: 512, K: 1024, N: 4096})
        expected = 2 * 512 * 1024 * 4096
        assert concrete == expected


# ==============================================================================
# Symbolic IR Node Tests
# ==============================================================================

class TestSymbolicBlockNode:
    """测试 BlockNode 的符号计算集成."""
    
    def test_blocknode_symbolic_attrs(self):
        """测试 BlockNode 使用符号属性."""
        H = Symbol('H', positive=True, integer=True)
        B, S = symbols('B S', positive=True, integer=True)
        
        linear = BlockNode(
            name="fc1",
            block_type="Linear",
            attrs={
                "in_features": H,
                "out_features": 4 * H,
                "batch_seq": B * S,
            }
        )
        
        flops = linear.compute_flops()
        assert isinstance(flops, Expr)
        
        # 代入具体值
        concrete = flops.subs({B: 4, S: 2048, H: 4096})
        assert concrete > 0
    
    def test_blocknode_symbolic_param_bytes(self):
        """测试 BlockNode 计算符号参数内存."""
        H = Symbol('H', positive=True, integer=True)
        
        linear = BlockNode(
            name="fc1",
            block_type="Linear",
            attrs={"in_features": H, "out_features": 4 * H}
        )
        
        param_bytes = linear.compute_param_bytes()
        assert isinstance(param_bytes, Expr)


class TestSymbolicOpNode:
    """测试 OpNode 的符号计算能力."""
    
    def test_opnode_symbolic_workload(self):
        """测试 OpNode 使用符号 workload."""
        M, K, N = symbols('M K N', positive=True, integer=True)
        
        op = OpNode(
            name="mm_0",
            op_type="Matmul",
            attrs={"M": M, "K": K, "N": N}
        )
        
        flops = op.compute_flops()
        assert isinstance(flops, Expr)
    
    def test_opnode_explicit_symbolic(self):
        """测试 OpNode 显式设置符号 workload."""
        T = Symbol('T', positive=True)
        
        op = OpNode(
            name="mm_0",
            op_type="Matmul",
            flops=2 * T,
            memory_bytes=4 * T,
        )
        
        assert op.compute_flops() == 2 * T
        assert op.compute_memory() == 4 * T


class TestSymbolicScheduledOp:
    """测试 ScheduledOp 的符号时序."""
    
    def test_scheduledop_symbolic_timing(self):
        """测试 ScheduledOp 使用符号时间."""
        T = Symbol('T', positive=True)
        
        op = OpNode(name="mm_0", op_type="Matmul")
        scheduled = ScheduledOp(
            op=op,
            start=T,
            duration=2 * T,
        )
        
        assert scheduled.start == T
        assert scheduled.duration == 2 * T
        assert scheduled.end == 3 * T


# ==============================================================================
# Symbolic GraphIR Tests
# ==============================================================================

class TestSymbolicGraphIR:
    """测试 GraphIR 的符号支持."""
    
    def test_graphir_symbols(self):
        """测试 GraphIR 符号表."""
        B = Symbol('B', positive=True, integer=True)
        S = Symbol('S', positive=True, integer=True)
        H = Symbol('H', positive=True, integer=True)
        
        graph = GraphIR(
            name="test",
            symbols={"B": B, "S": S, "H": H},
            metadata={"batch_size": B, "seq_len": S, "hidden": H}
        )
        
        assert "B" in graph.symbols
        assert "S" in graph.symbols
        assert "H" in graph.symbols
        
        # 元数据中可以使用符号
        assert graph.metadata["batch_size"] == B


# ==============================================================================
# SymMax Integration Tests
# ==============================================================================

class TestSymMaxIntegration:
    """测试 SymMax 与新 IR 系统的集成."""
    
    def setup_method(self):
        """每个测试前清理缓存."""
        clear_expr_cache()
    
    def test_symmax_in_scheduling(self):
        """测试 SymMax 用于调度时间计算."""
        T1, T2 = symbols('T1 T2', positive=True)
        
        # 模拟两个并行任务取 max
        end_time = SymMax(T1, T2)
        
        # 代入具体值
        result = end_time.eval({T1: 100, T2: 150})
        assert result == 150
        
        result = end_time.eval({T1: 200, T2: 150})
        assert result == 200
    
    def test_symmax_with_expr(self):
        """测试 SymMax 与 SymPy 表达式混合使用."""
        H = Symbol('H', positive=True, integer=True)
        
        # 计算两个 Linear 层的 FLOPs 取 max
        flops1 = 2 * H * 4 * H  # H -> 4H
        flops2 = 2 * 4 * H * H  # 4H -> H
        
        max_flops = SymMax(flops1, flops2)
        
        result = max_flops.eval({H: 4096})
        expected = max(2 * 4096 * 4 * 4096, 2 * 4 * 4096 * 4096)
        assert result == expected
    
    def test_symmax_arithmetic(self):
        """测试 SymMax 的算术运算."""
        A, B = symbols('A B', positive=True)
        
        m = SymMax(A, B)
        
        # SymMax + number
        result = m + 10
        assert result.eval({A: 5, B: 3}) == 15  # max(5,3) + 10
        
        # SymMax * number
        result = m * 2
        assert result.eval({A: 5, B: 3}) == 10  # max(5,3) * 2


# ==============================================================================
# DSL with Symbols Tests
# ==============================================================================

class TestDSLSymbolic:
    """测试 DSL 构建时使用符号."""
    
    def test_dsl_symbolic_attrs(self):
        """测试 DSL 使用符号属性."""
        H = Symbol('H', positive=True, integer=True)
        B = Symbol('B', positive=True, integer=True)
        S = Symbol('S', positive=True, integer=True)
        
        with Transformer("symbolic_model") as m:
            m.metadata(batch_size=B, seq_len=S, hidden=H)
            m.Linear("fc1", in_features=H, out_features=4*H, batch_seq=B*S)
            m.Linear("fc2", in_features=4*H, out_features=H, batch_seq=B*S)
        
        graph = m.build()
        
        # 验证符号被正确存储
        assert graph.metadata["batch_size"] == B
        assert graph.metadata["hidden"] == H
        
        # 验证 Block 属性中的符号
        fc1 = graph.root.children[0]
        assert fc1.attrs["in_features"] == H
        assert fc1.attrs["out_features"] == 4 * H
        
        # 计算 FLOPs 应该返回符号表达式
        flops = fc1.compute_flops()
        assert isinstance(flops, Expr)
    
    def test_dsl_symbolic_compute(self):
        """测试 DSL 构建后计算符号 workload."""
        H = Symbol('H', positive=True, integer=True)
        BS = Symbol('BS', positive=True, integer=True)
        
        with Transformer("model") as m:
            m.Linear("fc1", in_features=H, out_features=4*H, batch_seq=BS)
        
        graph = m.build()
        fc1 = graph.root.children[0]
        
        flops = fc1.compute_flops()
        
        # 代入具体值
        concrete = flops.subs({H: 4096, BS: 8192})
        expected = 2 * 8192 * 4096 * 4 * 4096 + 8192 * 4 * 4096
        assert concrete == expected


# ==============================================================================
# End-to-End Symbolic Test
# ==============================================================================

class TestSymbolicEndToEnd:
    """端到端符号计算测试."""
    
    def test_symbolic_model_to_schedule(self):
        """测试符号模型到 Schedule 的转换."""
        from src.blueprinting.ir.passes import ExpandPass
        
        H = Symbol('H', positive=True, integer=True)
        BS = Symbol('BS', positive=True, integer=True)
        
        # 构建符号模型
        with Transformer("symbolic") as m:
            m.metadata(hidden=H, batch_seq=BS)
            m.Linear("fc1", in_features=H, out_features=4*H, batch_seq=BS)
        
        graph = m.build()
        
        # 展开到 Schedule IR
        expand = ExpandPass()
        schedule = expand.run(graph)
        
        # 验证有 Op 生成
        assert schedule.total_ops > 0
        
        # Op 的 source_block 应该正确追踪
        ops = list(schedule.iter_ops())
        assert any("fc1" in (op.op.source_block or "") for op in ops)


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
