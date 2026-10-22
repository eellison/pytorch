# Runs the patched test file's CuTe tests with scripts/cute_arith_shim.py (or without it: --no-shim, to see the new test fail; --all: the whole class)
import importlib.util, os, sys, unittest
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shim = "--no-shim" not in sys.argv
sys.argv = [a for a in sys.argv if a != "--no-shim"]
import torch  # noqa: F401
if shim:
    sys.path.insert(0, os.path.join(R, "scripts"))
    import cute_arith_shim  # noqa: F401
spec = importlib.util.spec_from_file_location("tcute", os.path.join(R, "patches/cute_arith/b/test/test_cuda_host_trace_cute.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
names = sys.argv[1:] or ["test_an_identity_tensor_is_a_launch", "test_a_coordinate_tensors_basis_strides_are_static_leaves", "test_every_type_of_these_compiles_is_evaluated", "test_an_unsupported_op_is_an_eager_call_naming_it"]
if names == ["--all"]:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(m.TestHostTraceCute)
else:
    suite = unittest.TestSuite(m.TestHostTraceCute(n) for n in names)
r = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if r.wasSuccessful() else 1)
