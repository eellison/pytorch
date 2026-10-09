# The patched launcher (land/core/trtllm_cpp/src) compiled as FlashInfer compiles it, with the fi_ht aliases as the
# original types, is the stock code: same host functions (names and sizes) and the same instructions with immediates
# masked (the patch moves lines, and FLASHINFER_ERROR passes __LINE__), and the same device code. Also: src is what
# ht/make_patch.py writes from upstream/src, and upstream/src is the installed FlashInfer's (UPSTREAM.sha256).
# CPU only:
#   taskset -c 108-143 bash land/core/trtllm_cpp/python_trtllm_cpp.sh land/core/trtllm_cpp/tests/test_drift.py
import hashlib
import os
import re
import subprocess
import sys
import tempfile

from torch.testing._internal.common_utils import run_tests, TestCase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ht"))
import build_flags as bf  # noqa: E402

LAUNCHER = "csrc/trtllm_fmha_kernel_launcher.cu"


def compile_launcher(src_root: str, patched: bool, out: str) -> None:
    flags = bf.stock_cuda_flags(patched_first=patched)
    if patched:
        flags += [f"-I{os.path.join(ROOT, 'ht')}"]
    subprocess.run([bf.NVCC, "-c", os.path.join(src_root, LAUNCHER), "-o", out, *flags], check=True)


def _name(sym: str) -> str:
    # nvcc's per-translation-unit names carry a hash of the file (_GLOBAL__sub_I_..._003d3457_...)
    return re.sub(r"_[0-9a-f]{8}_[0-9a-f]{8}_\d+_", "_#_", sym)


def host_functions(obj: str) -> dict[str, int]:
    out = subprocess.run(["nm", "--defined-only", "-S", obj], capture_output=True, text=True, check=True).stdout
    funcs = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in "tTwW":
            funcs[_name(parts[3])] = int(parts[1], 16)
    return funcs


def masked_text(obj: str) -> list[str]:
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", obj], capture_output=True, text=True, check=True).stdout
    lines = []
    for line in out.splitlines():
        m = re.match(r"^\s+[0-9a-f]+:\s+(.*)$", line)
        if m:
            lines.append(re.sub(r"#-?0x[0-9a-f]+|#-?\d+|<[^>]*>|\b[0-9a-f]{4,}\b", "#", m.group(1)))
        elif line.endswith(">:"):
            lines.append(_name(line.split()[-1]))
    # an equality compare's operands may come in either order (TVM_FFI_ICHECK_EQ's after the patch's sym_size)
    for i in range(len(lines) - 1):
        m = re.match(r"^cmp\t(\w+), (\w+)$", lines[i])
        if m and re.match(r"^b\.(eq|ne)\b", lines[i + 1]):
            lines[i] = "cmp\t" + ", ".join(sorted(m.groups()))
    return lines


def sass(obj: str) -> list[str]:
    out = subprocess.run([os.path.join(bf.CUDA_HOME, "bin/cuobjdump"), "-sass", obj], capture_output=True, text=True,
                         check=True).stdout
    return [line.split("*/", 1)[-1].strip() for line in out.splitlines() if "/*" in line and not line.strip().startswith("//")]


class TestDrift(TestCase):
    def test_no_braced_container_lists(self):
        # nvcc 13.0's cudafe builds `auto s = std::vector<c10::SymInt>{a, b, c}` in a template with one element
        # (test_nvcc_vector_init.cu): no braced container list anywhere in the patched files or the shim
        sys.path.insert(0, os.path.join(ROOT, "ht"))
        pattern = re.compile(r"\b(?:std::vector|std::array|std::initializer_list|c10::SmallVector|SmallVector|DimVector|SymDimVector)"
                             r"\s*<[^;{}()]*>\s*(?:\w+\s*)?\{")
        roots = [os.path.join(ROOT, "src"), os.path.join(ROOT, "ht")]
        found = []
        for root in roots:
            for d, _, names in os.walk(root):
                for n in names:
                    if n.endswith((".h", ".cuh", ".cu", ".cpp")):
                        text = open(os.path.join(d, n)).read()
                        found += [f"{n}:{text.count(chr(10), 0, m.start()) + 1}" for m in pattern.finditer(text)]
        # braced lists passed to functions in the shim's templates: records are built with constructors
        for n in ("ht_trace.h", "ht_types.h"):
            text = open(os.path.join(ROOT, "ht", n)).read()
            found += [f"{n}:{text.count(chr(10), 0, m.start()) + 1}" for m in re.finditer(r"(push|emplace)_back\(\{", text)]
        self.assertEqual(found, [])

    def test_traced_shape_lists(self):
        import build as b

        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "traced_shapes")
            flags = [{"-std=c++17": "-std=c++20"}.get(f, f) for f in bf.stock_cuda_flags() if f != "-DPy_LIMITED_API=0x03090000"]
            t, tvm = b.torch_root(), os.path.join(os.path.dirname(bf.TVM_FFI_INCLUDE), "lib")
            subprocess.run([bf.NVCC, os.path.join(ROOT, "tests/traced_shapes.cu"), "-o", exe, f"-I{os.path.join(ROOT, 'ht')}", *flags,
                            *b.torch_includes(), f"-I{bf.SITE_DATA}/include/flashinfer/trtllm/fmha", "-Xcompiler", "-Wno-class-memaccess",
                            f"-L{t}/lib", "-lc10", "-lc10_cuda", f"-L{bf.CUDA_HOME}/lib64/stubs", "-lcuda", "-Xlinker", f"-rpath,{t}/lib",
                            f"-L{tvm}", "-ltvm_ffi", "-Xlinker", f"-rpath,{tvm}"], check=True)
            out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "Q 4 4 O 5 5 K 4 4")

    def test_upstream_is_installed_flashinfer(self):
        with open(os.path.join(ROOT, "src/UPSTREAM.sha256")) as f:
            for line in f:
                digest, rel = line.split()
                site = os.path.join(bf.SITE_DATA, rel)
                for path in (site, os.path.join(ROOT, "upstream/src", rel)):
                    with open(path, "rb") as g:
                        self.assertEqual(hashlib.sha256(g.read()).hexdigest(), digest, path)

    def test_src_is_make_patch_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, FI_HT_PATCH_OUT=tmp)
            subprocess.run([sys.executable, os.path.join(ROOT, "ht/make_patch.py")], check=True, env=env)
            for d, _, names in os.walk(tmp):
                for n in names:
                    rel = os.path.relpath(os.path.join(d, n), tmp)
                    with open(os.path.join(d, n)) as a, open(os.path.join(ROOT, "src", rel)) as b:
                        self.assertEqual(a.read(), b.read(), rel)

    def test_ordinary_build_is_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            stock, patched = os.path.join(tmp, "stock.o"), os.path.join(tmp, "patched.o")
            compile_launcher(bf.SITE_DATA, False, stock)
            compile_launcher(os.path.join(ROOT, "src"), True, patched)
            a, b = host_functions(stock), host_functions(patched)
            self.assertEqual(sorted(a), sorted(b))
            self.assertEqual({k: v for k, v in a.items() if b[k] != v}, {})
            ta, tb = masked_text(stock), masked_text(patched)
            self.assertGreater(len(ta), 1000)
            self.assertEqual(ta, tb)
            sa, sb = sass(stock), sass(patched)
            self.assertGreater(len(sa), 10)
            self.assertEqual(sa, sb)
            print(f"stock == patched: {len(a)} host functions, {len(ta)} instructions, {len(sa)} SASS lines")


if __name__ == "__main__":
    run_tests()
