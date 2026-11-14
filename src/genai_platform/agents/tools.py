"""Agent tools: safe AST calculator, Wikipedia lookup, and RAG search."""

from __future__ import annotations

import ast
import logging
import math
import operator
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[str], str]

    def __call__(self, arg: str) -> str:
        return self.fn(arg)


# ---------------------------------------------------------------------------
# Safe AST Calculator
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
}

_SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return float(_SAFE_CONSTANTS[node.id])
        raise ValueError(f"Unknown variable or constant: {node.id}")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPS:
            val = _eval_ast(node.operand)
            return float(_SAFE_OPS[op_type](val))
        raise ValueError(f"Unsupported unary operator: {op_type}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPS:
            left = _eval_ast(node.left)
            right = _eval_ast(node.right)
            if op_type is ast.Pow:
                # Cap exponent at 1000 to prevent DoS
                if abs(right) > 1000:
                    raise ValueError(f"Exponent too large (max 1000): {right}")
            return float(_SAFE_OPS[op_type](left, right))
        raise ValueError(f"Unsupported binary operator: {op_type}")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
            func = _SAFE_FUNCS[node.func.id]
            args = [_eval_ast(a) for a in node.args]
            return float(func(*args))
        raise ValueError("Disallowed function call")
    else:
        raise ValueError(f"Disallowed expression node: {type(node)}")


def safe_calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression using an AST whitelist (no eval)."""
    expr = expression.strip()
    if not expr:
        return "Empty calculation expression."
    # Replace ^ with ** for power
    expr = expr.replace("^", "**")
    try:
        parsed = ast.parse(expr, mode="eval")
        result = _eval_ast(parsed)
        # Format neatly
        if abs(result - round(result)) < 1e-12:
            return str(int(round(result)))
        return str(round(result, 8))
    except Exception as e:
        return f"Calculation error: {e}"


# ---------------------------------------------------------------------------
# Wikipedia lookup
# ---------------------------------------------------------------------------

def safe_wikipedia(query: str) -> str:
    """Look up a summary of a topic on Wikipedia."""
    q = query.strip().strip("\"'")
    if not q:
        return "Empty Wikipedia query."
    try:
        import wikipedia
        summary = wikipedia.summary(q, sentences=4, auto_suggest=True)
        return summary.strip()
    except Exception as e:
        err_type = type(e).__name__
        if "DisambiguationError" in err_type:
            options = getattr(e, "options", [])[:5]
            return f"Ambiguous topic '{q}'. Possible options: {', '.join(options)}"
        if "PageError" in err_type:
            return f"No Wikipedia page found for topic: '{q}'"
        return f"Wikipedia search error: {e}"


# ---------------------------------------------------------------------------
# RAG Search Tool
# ---------------------------------------------------------------------------

def make_rag_tool(retriever: Any, k: int = 4) -> Tool:
    """Create a RAG search tool from any retriever."""

    def _search(query: str) -> str:
        q = query.strip().strip("\"'")
        hits = retriever.retrieve(q, k=k)
        if not hits:
            return "No matching documents found in the knowledge base."
        lines = []
        for i, h in enumerate(hits, start=1):
            lines.append(f"[{i}] ({h.cite}): {h.text[:300]}")
        return "\n".join(lines)

    return Tool(
        name="rag_search",
        description="Search the local documents / knowledge base for relevant passages. Input: search query string.",
        fn=_search,
    )


def default_tools(retriever: Any | None = None) -> list[Tool]:
    """Return default suite of tools."""
    tools = [
        Tool(
            name="calculator",
            description="Perform mathematical calculations. Supports + - * / // % ** sqrt log exp sin cos tan pi e. Input: mathematical expression.",
            fn=safe_calculator,
        ),
        Tool(
            name="wikipedia",
            description="Search Wikipedia for encyclopedic facts, biographies, and definitions. Input: topic title.",
            fn=safe_wikipedia,
        ),
    ]
    if retriever:
        tools.append(make_rag_tool(retriever))
    return tools
