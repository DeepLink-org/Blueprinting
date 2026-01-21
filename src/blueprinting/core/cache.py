"""表达式求值缓存

提供表达式级别的求值缓存，避免重复计算。
"""
from typing import Any, Dict
from sympy import Expr


class ExprCache:
    """表达式求值缓存
    
    缓存 (表达式, 替换字典) -> 结果 的映射，避免重复计算。
    
    Examples
    --------
    >>> from sympy import Symbol
    >>> cache = ExprCache()
    >>> x = Symbol('x')
    >>> expr = x ** 2 + 2 * x + 1
    >>> cache.eval(expr, {x: 3})  # 计算并缓存
    16
    >>> cache.eval(expr, {x: 3})  # 从缓存获取
    16
    """
    
    def __init__(self, maxsize: int = 1024):
        """初始化缓存
        
        Args:
            maxsize: 最大缓存条目数，超过时使用 FIFO 淘汰
        """
        self._maxsize = maxsize
        self._cache: Dict[tuple, Any] = {}
    
    def _make_key(self, expr, subs: dict) -> tuple:
        """生成缓存键
        
        将表达式和替换字典转换为可哈希的键。
        """
        # 将 subs dict 转为可哈希的 frozenset
        subs_key = frozenset(subs.items())
        return (expr, subs_key)
    
    def eval(self, expr, subs: dict) -> Any:
        """求值表达式，使用缓存
        
        Args:
            expr: sympy 表达式
            subs: 符号替换字典 {Symbol: value}
            
        Returns:
            求值结果，如果是数值则转换为 float
        """
        key = self._make_key(expr, subs)
        
        if key not in self._cache:
            # 执行替换
            result = expr.subs(subs)
            
            # 尝试数值化
            if hasattr(result, 'is_number') and result.is_number:
                result = float(result)
            
            # 存入缓存
            self._cache[key] = result
            
            # FIFO 淘汰
            if len(self._cache) > self._maxsize:
                # 删除最早插入的条目
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
        
        return self._cache[key]
    
    def get(self, expr, subs: dict, default=None) -> Any:
        """获取缓存值，不存在时返回默认值"""
        key = self._make_key(expr, subs)
        return self._cache.get(key, default)
    
    def has(self, expr, subs: dict) -> bool:
        """检查是否有缓存"""
        key = self._make_key(expr, subs)
        return key in self._cache
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
    
    @property
    def size(self) -> int:
        """当前缓存条目数"""
        return len(self._cache)
    
    @property
    def maxsize(self) -> int:
        """最大缓存条目数"""
        return self._maxsize


# 全局缓存实例
expr_cache = ExprCache()
