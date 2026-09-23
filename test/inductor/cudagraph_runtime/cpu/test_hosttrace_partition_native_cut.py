# Owner(s): ["module: inductor"]

import gc
import weakref
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._inductor.runtime._cudagraph import hosttrace_partition as hp
from torch.cuda import _host_trace as ht
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestNativeCut(TestCase):
    def _tensor(self, root="p0", index=0, sizes=(4,), strides=(1,), offset=0):
        value = torch._C._CUDAGraphCutValue
        return torch._C._CUDAGraphCutTensor(
            root,
            index,
            [value(v, False) if type(v) is int else v for v in sizes],
            [value(v, False) if type(v) is int else v for v in strides],
            value(offset, False) if type(offset) is int else offset,
        )

    def _call(self, cut, box, boundary=None, values=(), known=None):
        result = cut(
            box,
            {} if boundary is None else boundary,
            values,
            set() if known is None else known,
        )
        self.assertIsNotNone(result)
        output, seconds = result
        self.assertIs(type(seconds), float)
        self.assertGreaterEqual(seconds, 0)
        return output

    @parametrize("location", ("input", "boundary"))
    def test_absolute_view_uses_current_storage_and_numeric_snapshot(self, location):
        value = torch._C._CUDAGraphCutValue
        reference = self._tensor(
            sizes=(value(0, True),), strides=(value(1, True),), offset=value(2, True)
        )
        cut = torch._C._CUDAGraphCut(
            torch.ops.aten.alias.default._handle, (reference,), {}
        )
        root = torch.empty(0)
        for start, shape, step, offset in (
            (0, 3, 2, 1),
            (100, 4, 3, 2),
            (200, 2, 4, 3),
        ):
            storage = torch.arange(start, start + 32, dtype=torch.float32)
            root.set_(storage.untyped_storage(), 7, (1,), (1,))
            known = set()
            actual = self._call(
                cut,
                [root] if location == "input" else [torch.zeros(1)],
                {} if location == "input" else {"p0": root},
                (shape, step, offset),
                known,
            )
            self.assertEqual(actual, storage.as_strided((shape,), (step,), offset))
            self.assertEqual(actual.stride(), (step,))
            self.assertEqual(actual.storage_offset(), offset)
            self.assertEqual(root.storage_offset(), 7)
            self.assertEqual(known, {storage.untyped_storage().data_ptr()})

    @parametrize("binding", ("default", "keyword"))
    def test_schema_default_and_keyword_scalar_mutate_once(self, binding):
        alpha = torch._C._CUDAGraphCutValue(0, True)
        kwargs = {} if binding == "default" else {"alpha": alpha}
        cut = torch._C._CUDAGraphCut(
            torch.ops.aten.add_.Tensor._handle,
            (self._tensor(), self._tensor("p1", 1)),
            kwargs,
        )
        destination, source = torch.ones(4), torch.full((4,), 2.0)
        expected = destination.clone()
        for multiplier in (1, 3, 2):
            known = set()
            actual = self._call(
                cut, [destination, source], values=(multiplier,), known=known
            )
            expected.add_(source, alpha=1 if binding == "default" else multiplier)
            self.assertEqual(actual, expected)
            self.assertEqual(destination, expected)
            self.assertEqual(source, torch.full_like(source, 2.0))
            self.assertEqual(
                known, {t.untyped_storage().data_ptr() for t in (destination, source)}
            )

    def test_schema_list_rebinds_without_retaining_caller_container(self):
        dimension = torch._C._CUDAGraphCutValue(0, True)
        dimensions = [dimension]
        cut = torch._C._CUDAGraphCut(
            torch.ops.aten.sum.dim_IntList._handle,
            (self._tensor(sizes=(2, 3), strides=(3, 1)), dimensions),
            {"keepdim": True, "dtype": torch.float64},
        )
        dimensions.clear()
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        for dim in (0, 1, 0):
            actual = self._call(cut, [source], values=(dim,))
            self.assertEqual(
                actual, source.sum(dim=[dim], keepdim=True, dtype=torch.float64)
            )
            self.assertEqual(actual.dtype, torch.float64)

    def test_tensor_list_reads_fresh_roots(self):
        tensors = [self._tensor(), self._tensor("p1", 1)]
        cut = torch._C._CUDAGraphCut(torch.ops.aten.cat.default._handle, (tensors,), {})
        tensors.reverse()
        for value in (1, 3, 2):
            roots = [torch.full((4,), float(value)), torch.full((4,), float(-value))]
            self.assertEqual(self._call(cut, roots), torch.cat(roots))

    def test_known_storage_is_recorded_before_set_rebinds_its_view(self):
        cut = torch._C._CUDAGraphCut(
            torch.ops.aten.set_.source_Tensor._handle,
            (self._tensor(), self._tensor("p1", 1)),
            {},
        )
        destination, source = torch.zeros(4), torch.arange(4, dtype=torch.float32)
        old_storage = destination.untyped_storage().data_ptr()
        known = set()
        actual = self._call(cut, [destination, source], known=known)
        self.assertEqual(actual, source)
        self.assertEqual(
            actual.untyped_storage().data_ptr(), source.untyped_storage().data_ptr()
        )
        self.assertEqual(destination, torch.zeros_like(destination))
        self.assertEqual(known, {old_storage, source.untyped_storage().data_ptr()})

    def test_second_view_out_of_bounds_fails_before_mutation(self):
        cut = torch._C._CUDAGraphCut(
            torch.ops.aten.add_.Tensor._handle,
            (self._tensor(), self._tensor("p1", 1, offset=2)),
            {},
        )
        destination, source = torch.ones(4), torch.full((4,), 2.0)
        known = set()
        with self.assertRaisesRegex(RuntimeError, "out of bounds"):
            cut([destination, source], {}, (), known)
        self.assertEqual(destination, torch.ones_like(destination))
        self.assertEqual(source, torch.full_like(source, 2.0))
        self.assertEqual(known, set())

    @parametrize("kind", ("missing", "bool", "wide"))
    def test_numeric_binding_error_precedes_mutation(self, kind):
        cut = torch._C._CUDAGraphCut(
            torch.ops.aten.add_.Tensor._handle,
            (self._tensor(), self._tensor("p1", 1)),
            {"alpha": torch._C._CUDAGraphCutValue(0, True)},
        )
        values, error = {
            "missing": ((), ValueError),
            "bool": ((True,), TypeError),
            "wide": ((1 << 64,), OverflowError),
        }[kind]
        roots = [torch.ones(4), torch.full((4,), 2.0)]
        known = set()
        with self.assertRaises(error):
            cut(roots, {}, values, known)
        self.assertEqual(roots[0], torch.ones_like(roots[0]))
        self.assertEqual(known, set())

    @parametrize("kind", ("duplicate", "unknown", "type"))
    def test_cold_schema_binding_rejects_invalid_arguments(self, kind):
        args = (self._tensor(), self._tensor("p1", 1))
        kwargs = {
            "duplicate": {"self": self._tensor()},
            "unknown": {"missing": 1},
            "type": {"alpha": "wrong"},
        }[kind]
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            torch._C._CUDAGraphCut(torch.ops.aten.add_.Tensor._handle, args, kwargs)

    def test_cold_raw_tensor_is_not_hidden_from_known_storage(self):
        with self.assertRaisesRegex(ValueError, "prepared root"):
            torch._C._CUDAGraphCut(
                torch.ops.aten.clone.default._handle, (torch.ones(4),), {}
            )

    def test_cold_nested_schema_list_is_unsupported(self):
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            torch._C._CUDAGraphCut(
                torch.ops.aten.sum.dim_IntList._handle, (self._tensor(), [[0]]), {}
            )

    def test_no_grad_call_releases_current_input(self):
        cut = torch._C._CUDAGraphCut(
            torch.ops.aten.clone.default._handle, (self._tensor(),), {}
        )
        with torch.enable_grad():
            source = torch.ones(4, requires_grad=True)
            reference = weakref.ref(source)
            actual = self._call(cut, [source])
            self.assertFalse(actual.requires_grad)
            self.assertTrue(torch.is_grad_enabled())
            self.assertEqual(actual, source)
            del source
        gc.collect()
        self.assertIsNone(reference())

    @parametrize("kind", ("missing", "subclass"))
    def test_unsupported_root_refuses_before_effects(self, kind):
        class TensorSubclass(torch.Tensor):
            pass

        cut = torch._C._CUDAGraphCut(
            torch.ops.aten.add_.Tensor._handle,
            (self._tensor(), self._tensor("p1", 1)),
            {},
        )
        destination = torch.ones(4)
        source = torch.full((4,), 2.0).as_subclass(TensorSubclass)
        known = set()
        roots = [destination] if kind == "missing" else [destination, source]
        self.assertIsNone(cut(roots, {}, (), known))
        self.assertEqual(destination, torch.ones_like(destination))
        self.assertEqual(known, set())


class TestPartitionNativeCut(TestCase):
    def _plan(self, operator, args, kwargs, output, symbols=()):
        partition = object.__new__(hp.Partition)
        partition.tape = SimpleNamespace(outputs=[])
        partition.uses = {}
        partition.out_plan = []
        partition.evaluator = ht._Evaluator()
        partition._input_index = {"p0": 0, "p1": 1}
        partition._alloc_nbytes = {}
        partition._boundary_indices = {
            value: index for index, value in enumerate(symbols)
        }
        op = ht._OpRec(0, operator, -1, 0, args, kwargs, 0, 0)
        op.seq[1] = 1
        op.outputs = output
        cut = hp._Cut(op, set())
        partition._plan_cut(0, cut)
        return partition, cut

    def test_native_cut_skips_input_materialization_and_keeps_output_checks(self):
        size, offset = sympy.symbols("size offset", integer=True)
        reference = hp._Ref("p0", (size,), (1,), offset, torch.float32)
        partition, cut = self._plan(
            torch.ops.aten.alias.default, (reference,), {}, reference, (size, offset)
        )
        with (
            mock.patch.object(
                partition,
                "_materialize",
                side_effect=AssertionError("Python input materialization"),
            ),
            mock.patch.object(hp, "tree_flatten", wraps=hp.tree_flatten) as flatten,
        ):
            for rows, displacement in ((3, 1), (4, 2), (2, 3)):
                source = torch.arange(16, dtype=torch.float32)
                actual = partition._run_cut(
                    0, cut, (rows, displacement), [source], {}, set()
                )
                self.assertIsNotNone(cut.native)
                self.assertEqual(
                    actual, [source.as_strided((rows,), (1,), displacement)]
                )
        self.assertEqual(flatten.call_count, 3)

    def test_unsupported_expression_uses_existing_python_path(self):
        stride = sympy.Symbol("stride", integer=True)
        expression = sympy.Abs(stride - 4)
        destination = hp._Ref("p0", (4,), (1,), 0, torch.float32)
        source = hp._Ref("p1", (4,), (1,), 0, torch.float32)
        partition, cut = self._plan(
            torch.ops.aten.add_.Tensor,
            (destination, source),
            {"alpha": expression},
            destination,
        )
        roots = [torch.ones(4), torch.full((4,), 2.0)]
        with mock.patch.object(
            partition, "_materialize", wraps=partition._materialize
        ) as materialize:
            actual = partition._run_cut(0, cut, {"stride": 3}, roots, {}, set())
        self.assertIsNone(cut.native)
        self.assertFalse(cut.native_ready)
        self.assertEqual(materialize.call_count, 3)
        self.assertEqual(actual, [torch.full((4,), 3.0)])

    def test_post_operator_output_mismatch_does_not_retry_mutation(self):
        destination = hp._Ref("p0", (4,), (1,), 0, torch.float32)
        source = hp._Ref("p1", (4,), (1,), 0, torch.float32)
        wrong_output = hp._Ref("p0", (3,), (1,), 0, torch.float32)
        partition, cut = self._plan(
            torch.ops.aten.add_.Tensor, (destination, source), {}, wrong_output
        )
        roots = [torch.ones(4), torch.full((4,), 2.0)]
        with mock.patch.object(
            partition, "_materialize", side_effect=AssertionError("Python retry")
        ):
            with self.assertRaisesRegex(hp.SwapMismatch, "expressions give"):
                partition._run_cut(0, cut, (), roots, {}, set())
        self.assertIsNotNone(cut.native)
        self.assertEqual(roots[0], torch.full((4,), 3.0))

    def test_native_root_refusal_falls_back_once(self):
        class TensorSubclass(torch.Tensor):
            pass

        destination = hp._Ref("p0", (4,), (1,), 0, torch.float32)
        source = hp._Ref("p1", (4,), (1,), 0, torch.float32)
        partition, cut = self._plan(
            torch.ops.aten.add_.Tensor, (destination, source), {}, destination
        )
        roots = [torch.ones(4), torch.full((4,), 2.0).as_subclass(TensorSubclass)]
        with mock.patch.object(
            partition, "_materialize", wraps=partition._materialize
        ) as materialize:
            actual = partition._run_cut(0, cut, (), roots, {}, set())
        self.assertIsNotNone(cut.native)
        self.assertEqual(materialize.call_count, 2)
        self.assertEqual(actual, [torch.full((4,), 3.0)])
        self.assertEqual(roots[0], torch.full((4,), 3.0))


if __name__ == "__main__":
    run_tests()
