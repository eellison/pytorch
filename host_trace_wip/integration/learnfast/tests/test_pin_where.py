# learnfast 10-15, pin rule (b) refinement: a guard recorded inside an ATen op's C++ symbolic evaluation is marked "aten",
# one from Python (here fn's own read of the size; in vLLM a library's int read) "python"; prov_redo exempts only
# pins marked aten alone. Run under diag27 with PROV_PIN_ATEN=1 (see run line in learnfast/STATUS.md).
import sys

import torch
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPinWhere(TestCase):
    def _where(self, f):
        dl = sys.modules["diag27_live"]
        env = f.variants[0].tape.shape_env
        return {repr(g): dl.GUARD_WHERE.get((id(env), i)) for i, (g, _) in enumerate(env.records)}

    def test_broadcast_size_one_check_is_aten(self):
        # x (n, 64) + y (1, 64): TensorIterator's broadcast compares n with 1 in C++ (a composite's host)
        def fn(x, y):
            return torch.ops.aten.expand_copy(y, x.shape) + x

        f = HostTraceReplay(fn)
        x, y = torch.randn(1, 64, device="cuda"), torch.randn(1, 64, device="cuda")
        for _ in range(3):  # a key's first call runs eagerly, the next traces
            self.assertEqual(f(x, y), fn(x, y))
        where = self._where(f)
        print("aten case", where)
        self.assertTrue(any(w == {"aten"} for w in where.values()))

    def test_python_size_read_is_python(self):
        def fn(x):
            return x * 2 if x.shape[0] == 1 else x * 3

        f = HostTraceReplay(fn)
        x = torch.randn(1, 64, device="cuda")
        for _ in range(3):
            self.assertEqual(f(x), fn(x))
        where = self._where(f)
        print("python case", where)
        # fn's own read of x.shape[0] == 1 is recorded from Python (an ATen host may record the same relation too): never
        # "aten" alone, so prov_redo keeps refusing a failing pin of it
        s0 = str(f.variants[0].tape.inputs[0].sizes[0])
        (w,) = [w for g, w in where.items() if g == f"(({s0} + -1) == 0)"]
        self.assertIn("python", w)


if __name__ == "__main__":
    run_tests()
