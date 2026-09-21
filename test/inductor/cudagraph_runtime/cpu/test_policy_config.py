"""Configuration snapshots retain the identity-bearing native runtime policy."""

import copy

from torch._inductor import config
from torch._inductor.runtime._cudagraph.api import NativeTerminalPolicy
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


class TestPolicyConfig(TestCase):
    @parametrize("config_snapshot", (False, True))
    def test_copy_preserves_policy_identity(self, config_snapshot):
        policy = NativeTerminalPolicy()
        self.addCleanup(policy.close)
        with config.patch(cudagraph_policy=policy):
            if config_snapshot:
                snapshot = config.get_config_copy()
                copied = snapshot["cudagraph_policy"]
            else:
                original = {"policy": policy, "same_policy": policy, "values": []}
                snapshot = copy.deepcopy(original)
                copied = snapshot["policy"]
                self.assertIs(snapshot["same_policy"], policy)
                self.assertIsNot(snapshot["values"], original["values"])
            self.assertIs(copied, policy)
            self.assertIs(config.cudagraph_policy, policy)
            self.assertEqual(copied.installations, ())
            copied.close()
            with self.assertRaisesRegex(RuntimeError, "Terminal policy is closed"):
                policy.wrap_output(None)


instantiate_parametrized_tests(TestPolicyConfig)

if __name__ == "__main__":
    run_tests()
