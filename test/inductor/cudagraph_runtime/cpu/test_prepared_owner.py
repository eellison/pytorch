"""Both frontends retain the same prepared replay and cleanup semantics."""

import gc
from weakref import ref



from torch._inductor.runtime._cudagraph import direct_host, policy, prepared
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


class Resource:
    def __init__(self, name, events, *, fail=False):
        self.name = name
        self.events = events
        self.fail = fail

    def close(self):
        self.events.append(self.name)
        if self.fail:
            raise RuntimeError(f"{self.name} close failed")


class TestPreparedOwner(TestCase):
    def release_variant(self, weak):
        variant = weak()
        if variant is not None:
            variant.entry.fail = False
            variant.program.fail = False
            variant.close()

    def test_frontends_share_type_and_retention(self):
        self.assertIs(policy.PreparedVariant, prepared.PreparedVariant)
        self.assertIs(direct_host.PreparedVariant, prepared.PreparedVariant)
        self.assertIs(policy._FAILED_VARIANTS, prepared._FAILED_VARIANTS)

    def test_close_orders_entry_before_program(self):
        events = []
        variant = prepared.PreparedVariant(Resource("entry", events), Resource("guard", events),
                                           Resource("program", events))
        variant.close()
        self.assertEqual(events, ["entry", "program"])
        self.assertFalse(any(item is variant for item in prepared._FAILED_VARIANTS))

    def test_successful_abort_preserves_original_error(self):
        events = []
        variant = prepared.PreparedVariant(Resource("entry", events), None, Resource("program", events))
        error = ValueError("preparation failed")
        error.add_note("original note")
        variant.abort(error)
        self.assertEqual(events, ["entry", "program"])
        self.assertEqual(error.__notes__, ["original note"])
        self.assertFalse(any(item is variant for item in policy._FAILED_VARIANTS))

    @parametrize("failure", ("entry", "program"))
    def test_failed_abort_retains_once_until_successful_close(self, failure):
        events = []
        entry = Resource("entry", events, fail=failure == "entry")
        program = Resource("program", events, fail=failure == "program")
        variant = direct_host.PreparedVariant(entry, None, program)
        weak = ref(variant)
        self.addCleanup(self.release_variant, weak)
        retained_list = policy._FAILED_VARIANTS
        error = ValueError("preparation failed")
        variant.abort(error)
        del variant
        gc.collect()
        retained = weak()
        self.assertIsNotNone(retained)
        self.assertEqual(sum(item is retained for item in retained_list), 1)
        expected_order = ["entry"] if failure == "entry" else ["entry", "program"]
        self.assertEqual(events, expected_order)
        retained.abort(error)
        self.assertEqual(events, expected_order * 2)
        self.assertEqual(sum(item is retained for item in retained_list), 1)
        self.assertEqual(error.__notes__,
                         [f"Terminal variant cleanup retained live resources: {failure} close failed"] * 2)
        entry.fail = program.fail = False
        retained.close()
        self.assertEqual(events[-2:], ["entry", "program"])
        self.assertIs(prepared._FAILED_VARIANTS, retained_list)
        self.assertIs(policy._FAILED_VARIANTS, retained_list)
        self.assertFalse(any(item is retained for item in retained_list))


instantiate_parametrized_tests(TestPreparedOwner)

if __name__ == "__main__":
    run_tests()
