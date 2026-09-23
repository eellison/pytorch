# Owner(s): ["module: cuda"]

from contextlib import ExitStack

from torch._inductor.runtime._cudagraph._sdk import activate


activate()

from cutlass._mlir import ir

from torch._inductor.runtime._cudagraph._compiler.argument_flow import (
    _check_host_operations,
    analyze_argument_flow,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


ROOTS = ("host", "cuda_num_binaries", "cuda_init", "cuda_load", "cuda_load_to_device")


@instantiate_parametrized_tests
class TestArgumentFlowScope(TestCase):
    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(ir.Context())
        stack.enter_context(ir.Location.unknown())
        stack.enter_context(ir.raw_values())

    def module(self, bodies=None, extra=""):
        bodies = {} if bodies is None else bodies
        source = "\n".join(
            f"llvm.func @{name}() {{ {bodies.get(name, '')} llvm.return }}"
            for name in ROOTS
        )
        module = ir.Module.parse(source + "\n" + extra)
        self.assertTrue(module.operation.verify())
        return module

    def registered(self, extra=""):
        loaders = "\n".join(
            f"""
          llvm.func @{name}(%libs: !llvm.ptr) -> i32 {{
            %one = llvm.mlir.constant(1 : i32) : i32
            %name = llvm.mlir.constant("kernel\00") : !llvm.array<7 x i8>
            %memory = llvm.alloca %one x !llvm.array<7 x i8> : (i32) -> !llvm.ptr
            llvm.store %name, %memory : !llvm.array<7 x i8>, !llvm.ptr
            %address = llvm.mlir.addressof @handle : !llvm.ptr
            %slot = llvm.getelementptr %libs[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
            %library = llvm.load %slot : !llvm.ptr -> !llvm.ptr
            %status = llvm.call @_cudaLibraryGetKernel(%address, %library, %memory) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
            llvm.return %status : i32
          }}
        """
            for name in ("cuda_load", "cuda_load_to_device")
        )
        source = """
          llvm.mlir.global internal constant @image("ABCD")
          llvm.mlir.global external @handle() : !llvm.ptr
          llvm.func @_cudaLibraryLoadData(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32) -> i32
          llvm.func @_cudaLibraryGetKernel(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
          llvm.func @_cudaLaunchKernelEx(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
          llvm.func @cuda_num_binaries() -> i32 {
            %one = llvm.mlir.constant(1 : i32) : i32
            llvm.return %one : i32
          }
          llvm.func @cuda_init(%libs: !llvm.ptr) -> i32 {
            %null = llvm.mlir.zero : !llvm.ptr
            %zero = llvm.mlir.constant(0 : i32) : i32
            %image = llvm.mlir.addressof @image : !llvm.ptr
            %slot = llvm.getelementptr %libs[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
            %status = llvm.call @_cudaLibraryLoadData(%slot, %image, %null, %null, %zero, %null, %null, %zero) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i32) -> i32
            llvm.return %status : i32
          }
          llvm.func @host() -> i32 {
            %address = llvm.mlir.addressof @handle : !llvm.ptr
            %handle = llvm.load %address : !llvm.ptr -> !llvm.ptr
            %zero = llvm.mlir.constant(0 : i32) : i32
            %parameters = llvm.alloca %zero x !llvm.ptr : (i32) -> !llvm.ptr
            %config = llvm.mlir.zero : !llvm.ptr
            %status = llvm.call @_cudaLaunchKernelEx(%config, %handle, %parameters) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
            llvm.return %status : i32
          }
        """
        module = ir.Module.parse(source + loaders + extra)
        self.assertTrue(module.operation.verify())
        return module, "host"

    def test_unrelated_sibling_external_call_is_not_selected(self):
        module = self.module(
            extra="""
            llvm.func @foreign_effect()
            llvm.func @sibling() { llvm.call @foreign_effect() : () -> () llvm.return }
        """
        )
        before = str(module)
        _check_host_operations(module, "host")
        self.assertEqual(str(module), before)
        with self.assertRaisesRegex(ValueError, "Unknown external host effect"):
            _check_host_operations(module, "sibling")

    @parametrize("root", ROOTS)
    def test_unknown_effect_reachable_from_each_root_is_rejected(self, root):
        module = self.module(
            {root: "llvm.call @helper() : () -> ()"},
            extra="""
            llvm.func @foreign_effect()
            llvm.func @helper() { llvm.call @foreign_effect() : () -> () llvm.return }
        """,
        )
        with self.assertRaisesRegex(ValueError, "Unknown external host effect"):
            _check_host_operations(module, "host")

    def test_shared_recursive_call_graph_is_finite_and_still_checked(self):
        module = self.module(
            {"host": "llvm.call @left() : () -> () llvm.call @right() : () -> ()"},
            extra="""
            llvm.func @left() { llvm.call @shared() : () -> () llvm.return }
            llvm.func @right() { llvm.call @shared() : () -> () llvm.return }
            llvm.func @shared() { llvm.call @left() : () -> () llvm.return }
        """,
        )
        _check_host_operations(module, "host")
        foreign = ir.Module.parse("llvm.func @foreign_effect()")
        with ir.InsertionPoint(module.body):
            next(iter(foreign.body.operations)).operation.clone()
        shared = next(
            view.operation
            for view in module.body.operations
            if view.operation.attributes["sym_name"].value == "shared"
        )
        call = next(iter(shared.regions[0].blocks[0].operations)).operation
        call.attributes["callee"] = ir.FlatSymbolRefAttr.get("foreign_effect")
        with self.assertRaisesRegex(ValueError, "Unknown external host effect"):
            _check_host_operations(module, "host")

    def test_reachable_indirect_call_is_rejected(self):
        module = self.module(
            {"host": "llvm.call @helper() : () -> ()"},
            extra="""
            llvm.func @helper() {
              %fn = llvm.mlir.zero : !llvm.ptr
              llvm.call %fn() : !llvm.ptr, () -> ()
              llvm.return
            }
        """,
        )
        with self.assertRaisesRegex(ValueError, "Unknown or indirect host call"):
            _check_host_operations(module, "host")

    def test_runtime_declaration_abi_check_is_preserved(self):
        module = self.module(extra="llvm.func @_cudaGetDevice(!llvm.ptr) -> i64")
        with self.assertRaisesRegex(ValueError, "unsupported emitted ABI"):
            _check_host_operations(module, "host")

    @parametrize("where", ("declaration", "call"))
    def test_runtime_calling_convention_check_is_preserved(self, where):
        module = self.module(
            {
                "host": """
            %ptr = llvm.mlir.zero : !llvm.ptr
            %result = llvm.call @_cudaGetDevice(%ptr) : (!llvm.ptr) -> i32
        """
            },
            extra="llvm.func @_cudaGetDevice(!llvm.ptr) -> i32",
        )
        function = next(
            view.operation
            for view in module.body.operations
            if view.operation.attributes["sym_name"].value
            == ("host" if where == "call" else "_cudaGetDevice")
        )
        operation = (
            next(
                view.operation
                for view in function.regions[0].blocks[0].operations
                if view.operation.name == "llvm.call"
            )
            if where == "call"
            else function
        )
        operation.attributes["CConv"] = ir.Attribute.parse("#llvm.cconv<fastcc>")
        with self.assertRaisesRegex(ValueError, "unsupported calling convention"):
            _check_host_operations(module, "host")

    def test_registration_and_whole_module_mutation_checks_survive(self):
        module, host = self.registered("""
            llvm.func @foreign_effect()
            llvm.func @sibling() { llvm.call @foreign_effect() : () -> () llvm.return }
        """)
        flow = analyze_argument_flow(module, host)
        self.assertGreater(len(flow.launches), 0)
        self.assertGreater(len(flow.binaries), 0)
        self.assertGreater(len(flow.registrations), 0)
        flow.check()
        sibling = next(
            view.operation
            for view in module.body.operations
            if view.operation.name == "llvm.func"
            and view.operation.attributes["sym_name"].value == "sibling"
        )
        sibling.attributes["changed"] = ir.UnitAttr.get()
        with self.assertRaisesRegex(RuntimeError, "Module changed"):
            flow.check()

    @parametrize("target", ("handle", "image"))
    def test_unreachable_registration_address_escape_is_still_rejected(self, target):
        module, host = self.registered()
        flow = analyze_argument_flow(module, host)
        name = (
            flow.registrations[0].handle_global
            if target == "handle"
            else flow.binaries[0].global_name
        )
        extra = f"""llvm.func @writer() {{
            %address = llvm.mlir.addressof @{name} : !llvm.ptr
            %byte = llvm.mlir.constant(1 : i8) : i8
            llvm.store %byte, %address : i8, !llvm.ptr
            llvm.return
        }}"""
        module, host = self.registered(extra)
        self.assertTrue(module.operation.verify())
        with self.assertRaisesRegex(ValueError, "writer or escaped address"):
            analyze_argument_flow(module, host)


if __name__ == "__main__":
    run_tests()
