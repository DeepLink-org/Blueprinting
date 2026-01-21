"""符号替换上下文

管理符号到数值的映射，与 hyperparameter.scope 协作。
"""
from contextlib import contextmanager
from typing import Any, Dict, Optional

import hyperparameter as hp
from sympy import Symbol

from .cache import expr_cache


class SubsContext:
    """符号替换上下文
    
    管理当前计算环境中的符号→数值映射，提供统一的求值接口。
    
    Examples
    --------
    手动绑定符号:
    >>> ctx = SubsContext()
    >>> HIDDEN = Symbol('hidden')
    >>> ctx.bind(HIDDEN, 4096)
    >>> ctx.eval(HIDDEN * 2)
    8192
    
    从 hp.scope 自动构建:
    >>> with SubsContext.from_scope() as ctx:
    ...     result = ctx.eval(expr)
    
    作为上下文管理器:
    >>> with SubsContext() as ctx:
    ...     ctx.bind(Symbol('x'), 10)
    ...     ctx.eval(Symbol('x') ** 2)
    100
    """
    
    def __init__(self):
        self._subs: Dict[Symbol, Any] = {}
    
    @property
    def subs(self) -> Dict[Symbol, Any]:
        """获取当前替换字典的副本"""
        return self._subs.copy()
    
    def bind(self, symbol: Symbol, value: Any) -> "SubsContext":
        """绑定符号到数值
        
        Args:
            symbol: sympy Symbol
            value: 对应的数值
            
        Returns:
            self，支持链式调用
        """
        self._subs[symbol] = value
        return self
    
    def bind_many(self, mappings: Dict[Symbol, Any]) -> "SubsContext":
        """批量绑定符号
        
        Args:
            mappings: {Symbol: value} 字典
            
        Returns:
            self，支持链式调用
        """
        self._subs.update(mappings)
        return self
    
    def unbind(self, symbol: Symbol) -> "SubsContext":
        """解除符号绑定
        
        Args:
            symbol: 要解绑的符号
            
        Returns:
            self，支持链式调用
        """
        self._subs.pop(symbol, None)
        return self
    
    def eval(self, expr, use_cache: bool = True) -> Any:
        """求值表达式
        
        Args:
            expr: sympy 表达式
            use_cache: 是否使用缓存，默认 True
            
        Returns:
            求值结果
        """
        if use_cache:
            return expr_cache.eval(expr, self._subs)
        
        result = expr.subs(self._subs)
        if hasattr(result, 'is_number') and result.is_number:
            result = float(result)
        return result
    
    def __enter__(self) -> "SubsContext":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    @classmethod
    @contextmanager
    def from_scope(cls, scope=None):
        """从 hp.scope 自动构建上下文
        
        自动绑定常用的模型和执行参数符号。
        
        Args:
            scope: hp.scope 对象，为 None 时使用当前 scope
            
        Yields:
            SubsContext 实例
        """
        ctx = cls()
        scope = scope or hp.scope()
        
        # 自动绑定模型参数符号
        model_params = [
            ('hidden', 'hidden'),
            ('feedforward', 'feedforward'),
            ('seqlen', 'seq_size'),
            ('attnheads', 'attn_heads'),
            ('attnsize', 'attn_size'),
            ('num_blocks', 'num_blocks'),
        ]
        
        for sym_name, attr_name in model_params:
            try:
                value = getattr(scope.model, attr_name) | 0
                if value:
                    ctx.bind(Symbol(sym_name), value)
            except (AttributeError, KeyError):
                pass
        
        # 自动绑定执行参数符号
        exe_params = [
            ('bsize', 'microbatch_size'),
            ('tpsize', 'tensor_par'),
            ('ppsize', 'pipeline_par'),
            ('dpsize', 'data_par'),
        ]
        
        for sym_name, attr_name in exe_params:
            try:
                value = getattr(scope.exe, attr_name) | 0
                if value:
                    ctx.bind(Symbol(sym_name), value)
            except (AttributeError, KeyError):
                pass
        
        yield ctx
