"""PROTOTYPE of fix (b) in DESIGN_nested_compile.md (lane-local overlay, no core edit; FN_WITNESS_INFRA=1).

A host trace's warm-up runs fn under the witness TorchDispatchMode (_host_trace_tape._Witness), and the trace call returns
that run's result. Dynamo does not compile under a dispatch mode unless the mode says it ignores compile internals
(torch/_dynamo/convert_frame.py, any_torch_dispatch_mode_on_stack): so a torch.compile'd function nested in an opaque op
(vLLM's CustomOp.maybe_compile'd forward_native inside vllm.moe_forward_shared) ran uncompiled in the warm-up, and the
trace call's output was not stock's. With ignore_compile_internals() True the mode is off while Dynamo looks the frame up /
compiles and on again for the compiled artifact's runtime (TorchDispatchMode.ignore_compile_internals), so the nested
function hits the same cache entry and runs the same compiled kernels as in stock; the witness still sees every operator
call those kernels' wrappers make. Nothing else of the witness changes."""
import torch.cuda._host_trace_tape as _tape


def install() -> None:
    if getattr(_tape._Witness, "_fn_witness_infra", False):
        return
    _tape._Witness.ignore_compile_internals = classmethod(lambda cls: True)
    _tape._Witness._fn_witness_infra = True
    print("FN_WITNESS_INFRA: the host-trace witness mode ignores compile internals (prototype of fix (b))", flush=True)
