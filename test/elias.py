import math
import operator
from torch._inductor.optimize_indexing import ValueRanges, ValueRangeAnalysis

def neg(x):
    return -x

def reciprocal(x):
    return 1 / x

def square(x):
    return x * x

unary = [
    (ValueRangeAnalysis.abs, abs),
    (ValueRangeAnalysis.ceil, math.ceil),
    (ValueRangeAnalysis.exp, math.exp),
    (ValueRangeAnalysis.floor, math.floor),
    (ValueRangeAnalysis.log, math.log),
    (ValueRangeAnalysis.neg, neg),
    (ValueRangeAnalysis.reciprocal, reciprocal),
    (ValueRangeAnalysis.sqrt, math.sqrt),
    (ValueRangeAnalysis.square, square),
]

binary = [
    (ValueRangeAnalysis.add, operator.__add__),
    (ValueRangeAnalysis.div, operator.__floordiv__),
    (ValueRangeAnalysis.maximum, max),
    (ValueRangeAnalysis.minimum, min),
    (ValueRangeAnalysis.mul, operator.__mul__),
    (ValueRangeAnalysis.pow, math.pow),
    (ValueRangeAnalysis.sub, operator.__sub__),
    (ValueRangeAnalysis.truediv, operator.__truediv__),
]

for (opr, op) in unary:
    for i in range(-9, 9):
        for j in range(i, 9):
            result_r = opr(ValueRanges(i, j))
            for ii in range(i, j+1):
                try:
                    result = op(ii)
                    if not result_r.fuzzy_contains(result):
                        tmp = ValueRanges(i, j)
                        breakpoint()
                        out = opr(tmp)
                        print('Buggy range: ', i, j, ii, result, result_r, opr.__name__)

                except:
                    continue

print('\nBinary')

for (opr, op) in binary:
    for i in range(-9, 9):
        for j in range(i, 9):
            for i2 in range(-9, 9):
                for j2 in range(i2, 9):
                    result_r = opr(ValueRanges(i, j), ValueRanges(i2, j2))
                    for ii in range(i, j+1):
                        for jj in range(i2, j2+1):
                            try:
                                result = op(ii, jj)
                                if not result_r.fuzzy_contains(result):
                                    print('Buggy range: ', i, j, i2, j2, ii, jj, result, result_r, opr.__name__)
                            except:
                                continue
