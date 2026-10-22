"""scripts/qsa_symrows.py: the rewritten QSA row-chunk loop yields the same (row_start, row_end) sequence as the original
`for row_start in range(0, rows, rows_per_chunk)` (CPU only; the substitution itself is asserted against the stock source).

    bash python_fn.sh tests/test_qsa_symrows.py
"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


def original(rows, rows_per_chunk):
    out = []
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        out.append((row_start, row_end))
    return out


def rewritten(rows, rows_per_chunk):
    out = []
    row_start = 0
    while row_start < rows:
        row_end = min(row_start + rows_per_chunk, rows)
        out.append((row_start, row_end))
        row_start += rows_per_chunk
    return out


class TestQsaSymrows(unittest.TestCase):
    def test_same_chunks(self):
        for rows_per_chunk in (1, 2, 127, 128, 2048):
            for rows in (0, 1, 127, 128, 129, 938, 5 * rows_per_chunk + 1):
                self.assertEqual(rewritten(rows, rows_per_chunk), original(rows, rows_per_chunk), (rows, rows_per_chunk))

    def test_substitution_applies_to_the_stock_source(self):
        import qsa_symrows

        src = qsa_symrows.patched_source(False)
        ast.parse(src)
        self.assertEqual(src.count("while row_start < rows:"), 1)
        self.assertEqual(src.count("row_start += rows_per_chunk"), 1)
        self.assertNotIn("for row_start in range(0, rows, rows_per_chunk)", src)  # the statement is gone (the comment names it)


if __name__ == "__main__":
    unittest.main()
