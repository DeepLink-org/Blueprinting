"""符号表达式扩展

提供扩展的符号表达式能力，作为模拟器的计算基础。
"""
from sympy import Expr, S, sympify


class SymPick(Expr):
    """符号条件表达式
    
    SymPick(condition, true_expr, false_expr)
    
    - 条件为符号时保持惰性求值
    - subs() 时自动选择分支
    - 可参与其他符号运算
    
    Examples
    --------
    >>> from sympy import Symbol
    >>> tp = Symbol('tp')
    >>> expr = SymPick(tp > 1, tp * 100, 0)
    >>> expr.subs(tp, 4)
    400
    >>> expr.subs(tp, 1)
    0
    
    嵌套使用:
    >>> sp = Symbol('sp')
    >>> nested = SymPick(tp > 1, SymPick(sp, tp * 2, tp), 0)
    >>> nested.subs([(tp, 4), (sp, True)])
    8
    """
    
    @property
    def cond(self):
        """条件表达式"""
        return self.args[0]
    
    @property
    def true_expr(self):
        """条件为真时的表达式"""
        return self.args[1]
    
    @property
    def false_expr(self):
        """条件为假时的表达式"""
        return self.args[2]
    
    def __new__(cls, cond, true_expr, false_expr):
        # 将参数转换为 sympy 对象
        cond = sympify(cond)
        true_expr = sympify(true_expr)
        false_expr = sympify(false_expr)
        
        # 条件可确定时直接返回结果
        if cond == S.true or cond is True:
            return true_expr
        if cond == S.false or cond is False:
            return false_expr
        
        # 尝试直接求值布尔条件
        try:
            if bool(cond):
                return true_expr
            return false_expr
        except TypeError:
            # 条件含符号，无法求值，保持 SymPick 形式
            pass
        
        return super().__new__(cls, cond, true_expr, false_expr)
    
    def _eval_subs(self, old, new):
        """替换后重新求值条件"""
        cond = self.cond.subs(old, new)
        true_expr = self.true_expr.subs(old, new)
        false_expr = self.false_expr.subs(old, new)
        
        # 尝试求值布尔条件
        try:
            if bool(cond):
                return true_expr
            return false_expr
        except TypeError:
            # 条件仍含符号，保持 SymPick 形式
            return SymPick(cond, true_expr, false_expr)
    
    def _sympystr(self, printer):
        """字符串表示"""
        return f"SymPick({printer.doprint(self.cond)}, {printer.doprint(self.true_expr)}, {printer.doprint(self.false_expr)})"
    
    def _latex(self, printer):
        """LaTeX 表示"""
        cond = printer.doprint(self.cond)
        t = printer.doprint(self.true_expr)
        f = printer.doprint(self.false_expr)
        return rf"\begin{{cases}} {t} & \text{{if }} {cond} \\ {f} & \text{{otherwise}} \end{{cases}}"
