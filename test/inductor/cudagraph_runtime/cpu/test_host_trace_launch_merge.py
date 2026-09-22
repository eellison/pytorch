# Owner(s): ["module: inductor"]

from types import SimpleNamespace

from torch.cuda import _host_trace_cute, _host_trace_triton
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestHostTraceLaunchMerge(TestCase):
    def _records(self):
        first, second = {"seq": 10}, {"seq": 50}
        return (
            {
                "launches": [first, second],
                "rng_slots": [
                    {"launch": 0, "offset": 24, "size": 8, "increment": 4},
                    {"launch": 1, "offset": 16, "size": 8, "increment": 8},
                    {"launch": 1, "offset": 48, "size": 4, "increment": 8},
                ],
                "written_roots": ["native"],
            },
            first,
            second,
        )

    def _merge(self, frontend, records, launches):
        trace = SimpleNamespace(
            launches=launches,
            written_roots=["native", frontend],
            observations=None,
            position=0,
        )
        tr = SimpleNamespace(triton=trace, cute=trace)
        module = _host_trace_triton if frontend == "triton" else _host_trace_cute
        module.merge(tr, records)

    def _check_slots(self, records, first, second):
        for slot, launch in zip(records["rng_slots"], (first, second, second)):
            self.assertIs(records["launches"][slot["launch"]], launch)
        self.assertEqual(
            [(s["offset"], s["size"], s["increment"]) for s in records["rng_slots"]],
            [(24, 8, 4), (16, 8, 8), (48, 4, 8)],
        )

    @parametrize("frontend", ("triton", "cute"))
    @parametrize("placement", ("before", "between", "after"))
    def test_rng_slots_keep_their_native_launch(self, frontend, placement):
        records, first, second = self._records()
        position, sequence = {"before": (0, 5), "between": (1, 30), "after": (2, 60)}[
            placement
        ]
        extra = {"seq": sequence}
        expected = [first, second]
        expected.insert(position, extra)
        self._merge(frontend, records, [extra])
        self.assertEqual(records["launches"], expected)
        self._check_slots(records, first, second)
        self.assertEqual(records["written_roots"], ["native", frontend])

    @parametrize("frontends", (("triton", "cute"), ("cute", "triton")))
    def test_sequential_merges_keep_the_original_rng_targets(self, frontends):
        records, first, second = self._records()
        triton = [{"seq": 0}, {"seq": 35}]
        cute = [{"seq": 20}, {"seq": 70}]
        launches = {"triton": triton, "cute": cute}
        for frontend in frontends:
            self._merge(frontend, records, launches[frontend])
            self._check_slots(records, first, second)
        self.assertEqual(
            records["launches"], [triton[0], first, cute[0], triton[1], second, cute[1]]
        )
        self.assertEqual([slot["launch"] for slot in records["rng_slots"]], [1, 4, 4])

    @parametrize("frontend", ("triton", "cute"))
    def test_dsl_only_launches_need_no_rng_slots(self, frontend):
        records = {"launches": [], "rng_slots": [], "written_roots": []}
        first, second = {"seq": 1}, {"seq": 2}
        self._merge(frontend, records, [first, second])
        self.assertIs(records["launches"][0], first)
        self.assertIs(records["launches"][1], second)
        self.assertEqual(records["rng_slots"], [])


if __name__ == "__main__":
    run_tests()
