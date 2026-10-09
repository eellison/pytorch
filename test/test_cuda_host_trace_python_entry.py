# Owner(s): ["module: cuda graphs"]
# The replay, opaque and harvest tests on HostTraceReplay's Python dispatch
# (torch.cuda._host_trace.cpp_entry = False), less the tests of what only the
# C++ entry does.

import os
import sys
from unittest import mock

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_cuda_host_trace_harvest as harvest  # noqa: E402
import test_cuda_host_trace_opaque as opaque  # noqa: E402
import test_cuda_host_trace_replay as replay  # noqa: E402


_CPP_ONLY = {
    # static_prefix, _HostTraceBound and their dirty records
    "TestStaticPrefix": None,
    "TestBoundCall": None,
    "TestDirtyRecords": ("test_alternating_statics_repatch_their_readers",),
    # a hit enters no Python, its native key and variant order
    "TestNativeReplay": (
        "test_a_graphsafe_rng_step_runs_no_python",
        "test_a_hit_runs_no_python",
        "test_a_host_seed_offset_step_runs_no_python",
        "test_an_index_with_a_none_runs_no_python",
        "test_an_aten_step_runs_no_python",
        "test_the_native_key_partitions_as_the_contract",
    ),
    "TestOpaqueCalls": (
        "test_a_new_key_at_a_keyed_site_is_a_native_row",
        "test_a_refused_key_is_a_row",
        "test_alternating_keys_take_their_rows",
        "test_keys_past_64_harvest",
    ),
    # the Python dispatch holds the arguments while it serves the call, so a
    # freed or boxed argument dies after the replay, not at its last use
    "TestReplayMemory": (
        "test_a_run_splits_past_a_peak_it_cannot_lower",
        "test_a_run_splits_where_a_freed_argument_dies",
        "test_a_slow_call_that_does_not_run_hands_the_call_back",
        "test_a_split_lowering_the_peak_by_little_is_undone",
        "test_call_boxed_drops_an_argument_after_its_last_use",
        "test_split_before_the_peak_allocation_first",
    ),
    # it compares the two paths itself
    "TestEntryParity": None,
}


def _python_entry(cls):
    skipped = _CPP_ONLY.get(cls.__name__, ())

    def setUp(self):
        if skipped is None or self._testMethodName.startswith(skipped):
            self.skipTest("a test of the C++ entry")
        patch = mock.patch.object(torch.cuda._host_trace, "cpp_entry", False)
        patch.start()
        self.addCleanup(patch.stop)
        cls.setUp(self)

    return type(f"{cls.__name__}PythonEntry", (cls,), {"setUp": setUp})


for _module in (replay, opaque, harvest):
    for _name, _cls in vars(_module).items():
        if isinstance(_cls, type) and issubclass(_cls, TestCase) and _cls.__module__ == _module.__name__:
            globals()[f"{_name}PythonEntry"] = _python_entry(_cls)


def setUpModule():
    from torch.cuda import _host_trace_hint_audit

    _host_trace_hint_audit.enable_for_tests()
    replay.setUpModule()


if __name__ == "__main__":
    run_tests()
