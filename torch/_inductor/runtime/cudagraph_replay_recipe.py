from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any


_BINARY = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
}
_COMPARE = {"eq": operator.eq, "le": operator.le, "ge": operator.ge}


@dataclass(frozen=True)
class IntExpr:
    op: str
    value: int | tuple[int, int] | None = None
    args: tuple[IntExpr, ...] = ()

    def evaluate(self, inputs: list[Any], depth: int = 0) -> int:
        if depth > 32 or type(self.args) is not tuple:
            raise ValueError("Unsupported integer expression structure")
        if self.op in ("constant", "boxed"):
            if type(self.value) is not int or self.args:
                raise ValueError("Integer leaves require exact integers")
            if self.op == "constant":
                result = self.value
            else:
                if not 0 <= self.value < len(inputs):
                    raise ValueError("Invalid boxed scalar index")
                result = inputs[self.value]
        elif self.op in ("size", "stride"):
            if (
                type(self.value) is not tuple or len(self.value) != 2 or self.args
                or any(type(value) is not int for value in self.value)
            ):
                raise ValueError("Tensor metadata leaves require a slot and dimension")
            slot, dimension = self.value
            if not 0 <= slot < len(inputs) or dimension not in (0, 1):
                raise ValueError("Unsupported tensor metadata index")
            tensor = inputs[slot]
            result = tensor.shape[dimension] if self.op == "size" else tensor.stride()[dimension]
        else:
            arity = 1 if self.op == "neg" else 2
            if self.value is not None or len(self.args) != arity or any(type(arg) is not IntExpr for arg in self.args):
                raise ValueError("Unsupported integer operation arguments")
            values = tuple(arg.evaluate(inputs, depth + 1) for arg in self.args)
            if self.op == "neg":
                result = -values[0]
            elif self.op in _BINARY:
                result = _BINARY[self.op](*values)
            else:
                raise ValueError("Unknown integer expression operation")
        if type(result) is not int or not -(2**63) <= result < 2**63:
            raise ValueError("Integer expression must produce an exact signed int64")
        return result


@dataclass(frozen=True)
class Guard:
    op: str
    left: IntExpr
    right: IntExpr

    def evaluate(self, inputs: list[Any]) -> bool:
        if self.op not in _COMPARE or type(self.left) is not IntExpr or type(self.right) is not IntExpr:
            raise ValueError("Unsupported recipe guard")
        return _COMPARE[self.op](self.left.evaluate(inputs), self.right.evaluate(inputs))

