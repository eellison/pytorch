# Owner(s): ["module: inductor"]
"""Kernel template bindings: a site's kernel nodes take their function, launch
configuration, argument image and attributes from a registered variant the numeric plan
selects per call; its memset nodes their destination, width and value. Every variant of
a site has the site's node chain (E28): node i of the variant is node i of the site, no
node is ever disabled, and a variant with another chain is refused at registration."""

import triton
import triton.language as tl

import torch
from torch._inductor.runtime._cudagraph.direct_hosttrace import _selected_variant
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@triton.jit(
    do_not_specialize=["value"], do_not_specialize_on_alignment=["value", "output"]
)
def store_value(value, output):
    tl.store(output, value)


@triton.jit(
    do_not_specialize=["value"], do_not_specialize_on_alignment=["value", "output"]
)
def store_double(value, output):
    tl.store(output, value * 2)


class TestKernelTemplateBinding(TestCase):
    def _kernel(self, kernel, stream, output):
        """(function, grid, block, shared, image, value slot, output slot) of a launch."""
        with torch.cuda.stream(stream):
            binary = kernel[(1,)](-(1 << 40), output)
            self.binaries.append(binary)
            slots = {
                name: index
                for index, (name, _) in enumerate(binary.src.signature.items())
            }
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                kernel[(1,)](-(1 << 40), output)
                node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[3][0][0]
            snapshot = graph._inspect_captured_kernel_nodes((node,))
        _, function, _, _, grid, block, shared, _, arguments = snapshot[3][0]
        image = bytearray(max(offset + width for offset, width, _ in arguments))
        for offset, width, data in arguments:
            image[offset : offset + width] = data
        attributes = tuple(torch._C._cuda_kernel_node_attributes(node))
        return (
            function,
            tuple(grid),
            tuple(block),
            shared,
            bytes(image),
            arguments[slots["value"]][0],
            arguments[slots["output"]][0],
            attributes,
        )

    def _row(self, kernel, value, attrs=None, grid=None, delta=0):
        function, base_grid, block, shared, image, value_offset, output_offset, base = (
            kernel
        )
        image = bytearray(image)
        image[value_offset : value_offset + 8] = int(value).to_bytes(
            8, "little", signed=True
        )
        image[output_offset : output_offset + 8] = bytes(8)
        return (
            function,
            base_grid if grid is None else grid,
            block,
            shared,
            bytes(image),
            ((output_offset, 0, delta),),
            (),
            base if attrs is None else attrs,
        )

    def _enabled(self, exec_, nodes):
        # the exec handle is read before the replay owner takes the graph
        from cuda.bindings import runtime

        from torch.cuda._utils import _check_cuda_bindings

        return [
            bool(_check_cuda_bindings(runtime.cudaGraphNodeGetEnabled(exec_, node)))
            for node in nodes
        ]

    def _entry(self, device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        self.binaries = []
        output = torch.empty((1,), dtype=torch.int64, device=device)
        single = self._kernel(store_value, stream, output)
        double = self._kernel(store_double, stream, output)
        site = torch._C._cuda_kernel_template_new_site()
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            nodes = []
            with torch.cuda.graph(graph, stream=stream):
                for row in (self._row(single, 1), self._row(single, 100)):
                    function, grid, block, shared, image, _, _, attrs = row
                    image = bytearray(image)
                    image[single[6] : single[6] + 8] = output.data_ptr().to_bytes(
                        8, "little"
                    )
                    torch._C._cuda_launch_kernel_image(
                        function,
                        grid,
                        block,
                        shared,
                        stream.cuda_stream,
                        bytes(image),
                        attrs,
                    )
                    nodes.append(
                        torch._C._cuda_get_capture_frontier(stream.cuda_stream)[3][0][0]
                    )
            graph.instantiate()
            self.exec = graph.raw_cuda_graph_exec()
            # the site's chain is two kernel nodes; every variant has two rows and the
            # second node's value is what the output reads. Variant 0 writes 100 then
            # 1; variant 1 changes the first node's function and writes 7 after it;
            # variant 2 launches both with a 2x1x1 cluster over a grid of 2, an
            # attribute the exec-level setter cannot carry, so the swap goes through
            # the graph node. A registered key is the key as select sees it: the
            # library settings appended
            settings = torch._C._cuda_kernel_template_library_settings()
            register = torch._C._cuda_kernel_template_register
            self.assertEqual(
                register(
                    site, [0, *settings], (self._row(single, 100), self._row(single, 1))
                ),
                0,
            )
            self.assertEqual(
                register(
                    site, [1, *settings], (self._row(double, 5), self._row(single, 7))
                ),
                1,
            )
            cluster = (2, 1, 1, *single[7][3:])
            clustered = self._row(single, 3, cluster, (2, 1, 1))
            self.assertEqual(register(site, [2, *settings], (clustered, clustered)), 2)
            # registering a key again returns its index
            self.assertEqual(
                register(
                    site, [0, *settings], (self._row(single, 9), self._row(single, 9))
                ),
                0,
            )
            # a variant with another node chain than the site's is refused
            with self.assertRaisesRegex(ValueError, "another node chain"):
                register(site, [7, *settings], (self._row(single, 1),))
            instructions = (
                ("constant", site),
                (
                    "call",
                    torch._C._cuda_kernel_template_selected_address(),
                    _selected_variant,
                    0,
                ),
            )
            batch = graph._prepare_kernel_replay_updates(
                (),
                1,
                (),
                (),
                len(instructions),
                template_bindings=((tuple(nodes), site, 1, ((0, None),), 0),),
            )
            entry = graph._make_boxed_replay(
                batch,
                stream,
                (output, *self.binaries),
                1,
                (0,),
                (),
                None,
                (("input", 0),),
                ((), instructions),
            )
        self.addCleanup(entry.close)
        self.addCleanup(stream.synchronize)
        self.nodes = nodes
        return entry, stream, site

    def test_variants_change_function_and_attributes(self, device):
        if torch.cuda.get_device_capability(device)[0] < 9:
            self.skipTest("the cluster attribute needs sm90 or later")
        entry, stream, site = self._entry(device)
        settings = len(torch._C._cuda_kernel_template_library_settings())
        with torch.cuda.stream(stream):
            output = torch.full((1,), -1, dtype=torch.int64, device=device)
            for key, expected in ((0, 1), (1, 7), (0, 1), (2, 3), (1, 7), (2, 3)):
                # the registry appends the library settings to the key it stores
                self.assertEqual(
                    torch._C._cuda_kernel_template_select(site, [key]), key
                )
                self.assertEqual(torch._C._cuda_kernel_template_selected(site), key)
                (result,) = entry([output])
                self.assertIs(result, output)
                self.assertEqual(int(output.item()), expected, f"key {key}")
            applies, graph_updates = entry._template_stats()
            self.assertEqual(applies, 6)
            self.assertGreaterEqual(graph_updates, 2)
            self.assertEqual(self._enabled(self.exec, self.nodes), [True, True])
            # a pointer move alone patches the slot without a swap
            other = torch.full((1,), -1, dtype=torch.int64, device=device)
            torch._C._cuda_kernel_template_select(site, [2])
            entry([other])
            self.assertEqual(int(other.item()), 3)
            self.assertEqual(entry._template_stats()[0], 6)
        # a select that misses records the key (with the settings) for the thread
        self.assertEqual(torch._C._cuda_kernel_template_select(site, [42]), -1)
        missed = torch._C._cuda_kernel_template_take_miss(site)
        self.assertEqual(len(missed), 1 + settings)
        self.assertEqual(missed[0], 42)
        self.assertIsNone(torch._C._cuda_kernel_template_take_miss(site))
        # replaying without a selection fails value validation before any work and
        # leaves the entry usable
        with torch.cuda.stream(stream):
            with self.assertRaisesRegex(
                ValueError, "outside its scalar ABI or grid range"
            ):
                entry([output])
            torch._C._cuda_kernel_template_select(site, [1])
            entry([output])
            self.assertEqual(int(output.item()), 7)

    def test_select_key_keeps_a_sites_last_hit_until_a_setting_changes(self, device):
        # the compiled predicates' select over (site, key pointer, key length): the
        # calling thread keeps the site's last hit under the settings epoch it was
        # selected under; a changed key or epoch runs the lookup, a miss is not kept
        import ctypes

        select_key = ctypes.CFUNCTYPE(
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
        )(torch._C._cuda_kernel_template_select_key_address())

        def select(site, *key):
            values = (ctypes.c_int64 * max(1, len(key)))(*key)
            got = select_key(site, values, len(key))
            self.assertEqual(torch._C._cuda_kernel_template_selected(site), got)
            return got

        stream = torch.cuda.Stream(device=device)
        self.binaries = []
        output = torch.empty((1,), dtype=torch.int64, device=device)
        row = self._row(self._kernel(store_value, stream, output), 1)
        site = torch._C._cuda_kernel_template_new_site()
        register = torch._C._cuda_kernel_template_register
        epoch = torch._C._cuda_kernel_template_settings_epoch
        take_miss = torch._C._cuda_kernel_template_take_miss
        settings = torch._C._cuda_kernel_template_library_settings()
        self.assertEqual(register(site, [3, 4, *settings], (row,)), 0)
        self.assertEqual(register(site, [3, 5, *settings], (row,)), 1)
        e0 = epoch()
        self.assertEqual(select(site, 3, 4), 0)
        self.assertEqual(select(site, 3, 4), 0)
        self.assertEqual(select(site, 3, 5), 1)
        self.assertEqual(select(site, 3, 4), 0)
        # a key of another length is another key: a miss, recorded with the settings,
        # and the hit before it is not what the next equal key returns
        self.assertEqual(select(site, 3), -1)
        self.assertEqual(take_miss(site), [3, *settings])
        self.assertEqual(select(site, 3, 4), 0)
        self.assertIsNone(take_miss(site))
        self.assertEqual(select(site + 1000, 3, 4), -1)
        self.assertEqual(epoch(), e0)
        # a setting the key hashes flips: the epoch moves, the same key misses (its
        # settings are new), a registration under the new settings serves, and the
        # flip back finds the first variant again
        matmul = torch.backends.cuda.matmul
        flag = matmul.allow_bf16_reduced_precision_reduction
        matmul.allow_bf16_reduced_precision_reduction = not flag
        try:
            self.assertGreater(epoch(), e0)
            flipped = torch._C._cuda_kernel_template_library_settings()
            self.assertNotEqual(flipped, settings)
            self.assertEqual(select(site, 3, 4), -1)
            self.assertEqual(take_miss(site), [3, 4, *flipped])
            self.assertEqual(register(site, [3, 4, *flipped], (row,)), 2)
            self.assertEqual(select(site, 3, 4), 2)
            self.assertEqual(select(site, 3, 4), 2)
            self.assertEqual(torch._C._cuda_kernel_template_select(site, [3, 4]), 2)
        finally:
            matmul.allow_bf16_reduced_precision_reduction = flag
        self.assertEqual(select(site, 3, 4), 0)
        self.assertEqual(torch._C._cuda_kernel_template_select(site, [3, 4]), 0)

    def test_every_setter_of_a_hashed_setting_bumps_the_epoch(self):
        # the settings the registry appends to every key (library_settings): each
        # of their setters moves the epoch, set to the value they already hold
        epoch = torch._C._cuda_kernel_template_settings_epoch
        matmul = torch.backends.cuda.matmul
        setters = {
            "allow_tf32": lambda: setattr(matmul, "allow_tf32", matmul.allow_tf32),
            "fp32_precision": lambda: setattr(
                matmul, "fp32_precision", matmul.fp32_precision
            ),
            "allow_fp16_reduced_precision_reduction": lambda: setattr(
                matmul,
                "allow_fp16_reduced_precision_reduction",
                matmul.allow_fp16_reduced_precision_reduction,
            ),
            "allow_bf16_reduced_precision_reduction": lambda: setattr(
                matmul,
                "allow_bf16_reduced_precision_reduction",
                matmul.allow_bf16_reduced_precision_reduction,
            ),
            "allow_fp16_accumulation": lambda: setattr(
                matmul, "allow_fp16_accumulation", matmul.allow_fp16_accumulation
            ),
            "use_deterministic_algorithms": lambda: torch.use_deterministic_algorithms(
                torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            ),
            "preferred_blas_library": lambda: torch.backends.cuda.preferred_blas_library(
                torch.backends.cuda.preferred_blas_library()
            ),
            "tunable.enable": lambda: torch.cuda.tunable.enable(
                torch.cuda.tunable.is_enabled()
            ),
            "tunable.tuning_enable": lambda: torch.cuda.tunable.tuning_enable(
                torch.cuda.tunable.tuning_is_enabled()
            ),
            "cublas workspace": lambda: torch._C._cuda_setCublasWorkspaceSize(
                torch._C._cuda_getCublasWorkspaceSize()
            ),
            "cublas workspace reset": torch._C._cuda_resetCublasWorkspaceSize,
            "cublaslt workspace": lambda: torch._C._cuda_setCublasLtWorkspaceSize(
                torch._C._cuda_getCublasLtWorkspaceSize()
            ),
            "cublaslt workspace reset": torch._C._cuda_resetCublasLtWorkspaceSize,
            "sm_carveout": lambda: torch._C._set_sm_carveout_experimental(
                torch._C._get_sm_carveout_experimental()
            ),
        }
        # the legacy allow_tf32 setter writes "ieee" where the new API may hold "none"
        # (the same kernels, another key value): the precision string is restored
        settings = torch._C._cuda_kernel_template_library_settings()
        precision = matmul.fp32_precision
        try:
            for name, setter in setters.items():
                before = epoch()
                setter()
                self.assertGreater(epoch(), before, name)
                matmul.fp32_precision = precision
        finally:
            matmul.fp32_precision = precision
        self.assertEqual(torch._C._cuda_kernel_template_library_settings(), settings)

    def test_variants_drive_memset_nodes(self, device):
        from cuda.bindings import runtime

        from torch.cuda._utils import _check_cuda_bindings

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        self.binaries = []
        # the site's chain: a memset of the operand's first element, then a kernel
        # writing its second (slot delta 8); every variant has that chain
        output = torch.full((2,), -1, dtype=torch.int64, device=device)
        other = torch.full((1,), -1, dtype=torch.int64, device=device)
        single = self._kernel(store_value, stream, output)
        site = torch._C._cuda_kernel_template_new_site()
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                _check_cuda_bindings(
                    runtime.cudaMemsetAsync(output.data_ptr(), 0, 8, stream.cuda_stream)
                )
                memset_node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[
                    3
                ][0][0]
                function, grid, block, shared, image, _, _, attrs = self._row(single, 1)
                image = bytearray(image)
                image[single[6] : single[6] + 8] = (output.data_ptr() + 8).to_bytes(
                    8, "little"
                )
                torch._C._cuda_launch_kernel_image(
                    function,
                    grid,
                    block,
                    shared,
                    stream.cuda_stream,
                    bytes(image),
                    attrs,
                )
                kernel_node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[
                    3
                ][0][0]
            graph.instantiate()
            exec_ = graph.raw_cuda_graph_exec()
            settings = torch._C._cuda_kernel_template_library_settings()
            register = torch._C._cuda_kernel_template_register
            # a memset row: ("memset", operand or None, delta or address, element size,
            # width, value); row i is node i of the site's chain. Variant 0: the memset
            # fills the operand's first 8 bytes with 0x11 and the kernel writes 7 to the
            # second; 1: a zeroing memset, then 3; 2: the memset at a fixed address
            # (another tensor's) and 5
            fill = int.from_bytes(bytes([0x11]) * 8, "little")
            row = lambda value: self._row(single, value, delta=8)  # noqa: E731
            self.assertEqual(
                register(site, [0, *settings], (("memset", 0, 0, 1, 8, 0x11), row(7))),
                0,
            )
            self.assertEqual(
                register(site, [1, *settings], (("memset", 0, 0, 1, 8, 0), row(3))), 1
            )
            self.assertEqual(
                register(
                    site,
                    [2, *settings],
                    (("memset", None, other.data_ptr(), 1, 8, 0x22), row(5)),
                ),
                2,
            )
            with self.assertRaisesRegex(ValueError, "another node chain"):
                register(site, [4, *settings], (row(3),))
            with self.assertRaisesRegex(ValueError, "another node chain"):
                register(site, [5, *settings], (row(3), ("memset", 0, 0, 1, 8, 0)))
            with self.assertRaisesRegex(ValueError, "byte memsets"):
                register(site, [6, *settings], (("memset", 0, 0, 4, 2, 0), row(3)))
            instructions = (
                ("constant", site),
                (
                    "call",
                    torch._C._cuda_kernel_template_selected_address(),
                    _selected_variant,
                    0,
                ),
            )
            batch = graph._prepare_kernel_replay_updates(
                (),
                1,
                (),
                (),
                len(instructions),
                template_bindings=(
                    ((memset_node, kernel_node), site, 1, ((0, None),), 0),
                ),
            )
            entry = graph._make_boxed_replay(
                batch,
                stream,
                (output, *self.binaries),
                1,
                (0,),
                (),
                None,
                (("input", 0),),
                ((), instructions),
            )
            self.addCleanup(entry.close)
            expected = {0: (fill, 7), 1: (0, 3), 2: (-1, 5)}
            for key in (0, 1, 2, 0, 2, 1):
                output.fill_(-1)
                torch._C._cuda_kernel_template_select(site, [key])
                entry([output])
                self.assertEqual(tuple(output.tolist()), expected[key], f"key {key}")
                if key == 2:
                    self.assertEqual(
                        int(other.item()), int.from_bytes(bytes([0x22]) * 8, "little")
                    )
                # no node of the exec is ever disabled
                self.assertEqual(
                    self._enabled(exec_, (memset_node, kernel_node)), [True, True]
                )
            self.assertEqual(entry._template_stats()[0], 6)
            # the memset's operand destination follows a pointer move without a swap
            moved = torch.full((2,), -1, dtype=torch.int64, device=device)
            torch._C._cuda_kernel_template_select(site, [1])
            entry([moved])
            self.assertEqual(tuple(moved.tolist()), (0, 3))
            self.assertEqual(tuple(output.tolist()), (0, 3))
            self.assertEqual(entry._template_stats()[0], 6)
        stream.synchronize()

    def test_template_binding_rejects_bound_and_missing_nodes(self, device):
        stream = torch.cuda.Stream(device=device)
        self.binaries = []
        output = torch.empty((1,), dtype=torch.int64, device=device)
        single = self._kernel(store_value, stream, output)
        with torch.cuda.stream(stream):
            binary = store_value[(1,)](-(1 << 40), output)
            slots = {
                name: index
                for index, (name, _) in enumerate(binary.src.signature.items())
            }
            stream.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=stream):
                store_value[(1,)](-(1 << 40), output)
                node = torch._C._cuda_get_capture_frontier(stream.cuda_stream)[3][0][0]
            graph.instantiate()
            site = torch._C._cuda_kernel_template_new_site()
            with self.assertRaisesRegex(ValueError, "names a bound kernel node"):
                graph._prepare_kernel_replay_updates(
                    ((node, slots["output"], 0),),
                    1,
                    (),
                    (),
                    2,
                    template_bindings=(((node,), site, 1, ((0, None),), 0),),
                )
            with self.assertRaisesRegex(ValueError, "needs kernel nodes"):
                graph._prepare_kernel_replay_updates(
                    (),
                    1,
                    (),
                    (),
                    2,
                    template_bindings=(((), site, 1, ((0, None),), 0),),
                )
            with self.assertRaisesRegex(IndexError, "Unknown kernel template site"):
                torch._C._cuda_kernel_template_register(
                    1 << 40, [0], (self._row(single, 1),)
                )
            # a binding whose nodes are not the chain its site's variants have
            two = torch._C._cuda_kernel_template_new_site()
            torch._C._cuda_kernel_template_register(
                two, [0], (self._row(single, 1), self._row(single, 2))
            )
            with self.assertRaisesRegex(ValueError, "not its site's chain"):
                graph._prepare_kernel_replay_updates(
                    (),
                    1,
                    (),
                    (),
                    2,
                    template_bindings=(((node,), two, 1, ((0, None),), 0),),
                )
        stream.synchronize()


instantiate_device_type_tests(TestKernelTemplateBinding, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
