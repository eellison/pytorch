# Owner(s): ["module: inductor"]

import os
import subprocess
import sys

from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase


class TestAutotuneQueueConfig(TestCase):
    def test_autotune_queue_config_aliases(self):
        with config.patch(
            {
                "autotune_queue": True,
                "autotune_queue_policy": "all",
                "autotune_queue_min_kernels": 7,
                "autotune_queue_max_live_bytes": 1234,
                "autotune_queue_max_live_kernels": 11,
                "autotune_queue_max_frontier_candidates": 13,
            }
        ):
            for name, expected in (
                ("coordinate_descent_tuning_batch", True),
                ("coordinate_descent_tuning_batch_policy", "all"),
                ("coordinate_descent_tuning_batch_min_kernels", 7),
                ("coordinate_descent_tuning_batch_max_live_bytes", 1234),
                ("coordinate_descent_tuning_batch_max_live_kernels", 11),
                ("coordinate_descent_tuning_batch_max_frontier_candidates", 13),
            ):
                self.assertEqual(getattr(config, name), expected)

        with config.patch(
            {
                "coordinate_descent_tuning_batch": True,
                "coordinate_descent_tuning_batch_policy": "auto",
                "coordinate_descent_tuning_batch_min_kernels": 5,
            }
        ):
            for name, expected in (
                ("autotune_queue", True),
                ("autotune_queue_policy", "auto"),
                ("autotune_queue_min_kernels", 5),
            ):
                self.assertEqual(getattr(config, name), expected)

    def test_autotune_queue_aliases_skip_portable_config_duplication(self):
        with config.patch(
            {
                "autotune_queue": True,
                "autotune_queue_policy": "all",
            }
        ):
            portable = config.save_config_portable(ignore_private_configs=False)

        for name in ("autotune_queue", "autotune_queue_policy"):
            self.assertNotIn(name, portable)
        self.assertTrue(portable["coordinate_descent_tuning_batch"])
        self.assertEqual(portable["coordinate_descent_tuning_batch_policy"], "all")

    def test_autotune_queue_env_aliases(self):
        env_names = [
            f"TORCHINDUCTOR_{prefix}{suffix}"
            for suffix in (
                "",
                "_POLICY",
                "_MIN_KERNELS",
                "_MAX_LIVE_BYTES",
                "_MAX_LIVE_KERNELS",
                "_MAX_FRONTIER_CANDIDATES",
            )
            for prefix in ("AUTOTUNE_QUEUE", "COORDINATE_DESCENT_TUNING_BATCH")
        ]

        def child_config(env_updates):
            env = os.environ.copy()
            for name in env_names:
                env.pop(name, None)
            env.update(env_updates)
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "\n".join(
                        [
                            "import torch._inductor.config as c",
                            "print(",
                            "    c.autotune_queue,",
                            "    c.autotune_queue_policy,",
                            "    c.autotune_queue_min_kernels,",
                            "    c.autotune_queue_max_live_bytes,",
                            "    c.autotune_queue_max_live_kernels,",
                            "    c.autotune_queue_max_frontier_candidates,",
                            ")",
                        ]
                    ),
                ],
                env=env,
                text=True,
            )
            return output.strip().splitlines()[-1]

        cases = [
            (
                {
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH": "1",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH_POLICY": "all",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH_MIN_KERNELS": "4",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH_MAX_LIVE_BYTES": "5",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH_MAX_LIVE_KERNELS": "6",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH_MAX_FRONTIER_CANDIDATES": "7",
                },
                "True all 4 5 6 7",
            ),
            (
                {
                    "TORCHINDUCTOR_AUTOTUNE_QUEUE": "1",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH": "0",
                    "TORCHINDUCTOR_AUTOTUNE_QUEUE_POLICY": "auto",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH_POLICY": "none",
                    "TORCHINDUCTOR_AUTOTUNE_QUEUE_MIN_KERNELS": "8",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH_MIN_KERNELS": "1",
                    "TORCHINDUCTOR_AUTOTUNE_QUEUE_MAX_LIVE_BYTES": "9",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH_MAX_LIVE_BYTES": "1",
                    "TORCHINDUCTOR_AUTOTUNE_QUEUE_MAX_LIVE_KERNELS": "10",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH_MAX_LIVE_KERNELS": "1",
                    "TORCHINDUCTOR_AUTOTUNE_QUEUE_MAX_FRONTIER_CANDIDATES": "11",
                    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING_BATCH_MAX_FRONTIER_CANDIDATES": "1",
                },
                "True auto 8 9 10 11",
            ),
        ]
        for env_updates, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(child_config(env_updates), expected)

if __name__ == "__main__":
    run_tests()
