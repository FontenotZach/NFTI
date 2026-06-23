import ast
import math
from typing import Any, Dict


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    result = float(value)
    if math.isnan(result):
        return float("nan")
    return result


def safe_eval_arithmetic(expr: str, variables: Dict[str, Any]) -> float:
    """
    Safely evaluate a simple arithmetic expression containing:
    - numeric literals
    - variable names (looked up in `variables`)
    - +, -, *, / and parentheses

    Anything else raises ValueError.
    """

    # Parse once as an AST expression.
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Python 3.8+ uses Constant for literals.
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Disallowed literal: {node.value!r}")

        # Variable lookup.
        if isinstance(node, ast.Name):
            if node.id in variables:
                return _to_float(variables[node.id])
            return float("nan")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BINOPS:
                raise ValueError(f"Disallowed operator: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            if math.isnan(left) or math.isnan(right):
                return float("nan")
            return _ALLOWED_BINOPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARYOPS:
                raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
            operand = _eval(node.operand)
            if math.isnan(operand):
                return float("nan")
            return _ALLOWED_UNARYOPS[op_type](operand)

        raise ValueError(f"Disallowed expression element: {type(node).__name__}")

    result = _eval(tree)
    return float(result)

