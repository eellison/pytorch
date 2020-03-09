import os
import sys

import torch
from torch.nn import functional as F
from torch.testing import FileCheck
import torchvision
from torch.testing._internal.common_nn import module_tests, new_module_tests
from functools import wraps
import warnings
from torch.testing._internal.common_methods_invocations import create_input
from copy import deepcopy
from torch.testing._internal.common_methods_invocations import unpack_variables


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, freeze_rng_state, enable_profiling_mode

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestFunctionalBlocks(JitTestCase):
    def test_subgraph_creation(self):
        def fn(x, y, z):
            x = x + 1
            y = y + 1
            z = z + 1
            z.add_(2)
            z = z * z
            y = y * z
            if y < 2:
                y = y + 5
            return x + y + z

        graph = torch.jit.script(fn).graph
        self.run_pass('create_functional_graphs', graph)

        # all uses of x and y should be sunk
        FileCheck().check(r"%x").check_not(r"%x").check("FunctionalGraph").check(r"%x").run(graph)
        FileCheck().check(r"%y").check_not(r"%y").check("FunctionalGraph").check(r"%y").run(graph)

        # Don't allow any outputs which escape scope, so there is one final addition in the graph
        FileCheck().check("Tensor = prim::Functional").check_next("aten::add").run(graph)

        # z + 1, z.add_(2) considered non functional, z = z * z should be considered functional
        FileCheck().check("add").check("add_").check_not("mul").check("FunctionalGraph").run(graph)

    def test_lower_linear(self):
        # linear is one of main use cases of removing mutation so add test so it doesnt regress
        @torch.jit.script
        def foo(x):
            return F.linear(x, torch.randn(20, 20), torch.randn(20))

        self.run_pass('inline', foo.graph)
        self.run_pass('peephole', foo.graph)
        self.run_pass('constant_propagation', foo.graph)
        FileCheck().check("aten::add_").run(foo.graph)
        input = torch.randn(20, 20)
        with freeze_rng_state():
            out1 = foo(input)

        self.run_pass('remove_mutation', foo.graph)
        FileCheck().check_not("aten::add_").run(foo.graph)
        with freeze_rng_state():
            out2 = foo(input)
        self.assertEqual(out1, out2)

    def test_remove_mutation_aten_inplace(self):
        def test_not_new_alias(x):
            y = x[0]
            y.add_(2)
            return y

        fn = torch.jit.script(test_not_new_alias)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check("aten::add_").run(graph)
        self.assertEqual(fn(torch.ones([2, 2])), test_not_new_alias(torch.ones([2, 2])))

        def test_no_lowering():
            x = torch.tensor([2, 2])
            x[0] = 3
            return x

        # there is no functional equivalent of x[0] = ...
        fn = torch.jit.script(test_no_lowering)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check("aten::copy_").run(graph)
        self.assertEqual(fn(), test_no_lowering())

        def test_move_before_not_valid():
            y = torch.tensor([2, 2])
            z = y + 2
            y.add_(2)
            return y, z

        fn = torch.jit.script(test_move_before_not_valid)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check("aten::add_").run(graph)
        self.assertEqual(fn(), test_move_before_not_valid())

        def test_successful():
            x = torch.tensor([2, 2])
            x.add_(1)
            x.add_(3)
            y = x + 4
            return x, y

        fn = torch.jit.script(test_successful)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check_not("aten::add_").run(graph)
        self.assertEqual(test_successful(), fn())

        def test_intermediary_use():
            x = torch.tensor([2, 2])
            x.add_(1)
            y = x + 4
            x.add_(3)
            return x, y

        fn = torch.jit.script(test_intermediary_use)
        graph = fn.graph
        FileCheck().check_count("aten::add_", 2).run(graph)
        self.run_pass('remove_mutation', graph)
        # Unable to remove the second add_ because of the y = x + 4 use
        # In the future we could duplicating the value of x as a temporary and replacing
        # its intermediary use (so long as aliasing is safe)
        FileCheck().check_count("aten::add_", 1).run(graph)
        self.assertEqual(test_intermediary_use(), fn())

    def test_remove_mutation_lists_append(self):
        def successful_remove():
            return [i for i in range(5)]  # noqa: C416

        fn = torch.jit.script(successful_remove)
        graph = fn.graph
        self.run_pass('loop_unrolling', graph)
        self.run_pass('remove_mutation', graph)
        self.run_pass('constant_propagation', graph)
        FileCheck().check("graph").check_next("Constant").check_next("return").run(graph)
        self.assertEqual(successful_remove(), successful_remove())

        def intermediary_use():
            a = [1, 2]
            b = len(a)
            a.append(3)
            return a

        fn = torch.jit.script(intermediary_use)
        graph = fn.graph
        FileCheck().check("append").run(graph)
        self.run_pass('remove_mutation', graph)
        # it is possible to remove the append here but don't currently have the logic for it
        FileCheck().check_not("append").run(graph)
        self.assertEqual(intermediary_use(), fn())

    def test_elias(self):
        m = torch.jit.script(torchvision.models.resnext50_32x4d().eval())
        m_f = m
        # m_f = torch._C._freeze_module(m._c)
        self.run_pass('loop_unrolling', m_f.forward.graph)
        self.run_pass('remove_mutation', m_f.forward.graph)
        self.run_pass('constant_propagation', m_f.forward.graph)
        input_shape = (1, 3, 224, 224)
        x = torch.rand(input_shape)
        with enable_profiling_mode():
            for _ in range(10):
                m_f.forward(x)
        self.run_pass('create_functional_graphs', torch.jit.last_executed_optimized_graph())

        print(m_f.forward)

    def test_nano_sparse(self):
        from torch import nn

        class NanoSparseNN(torch.nn.Module):
            def __init__(self):
                super(NanoSparseNN, self).__init__()
                self.emb = nn.EmbeddingBag(10000, 100)
                self.fc1 = nn.Linear(100, 1)

            def forward(self, ids, offsets):
                x = self.emb(ids, offsets)
                x = self.fc1(x)
                return nn.functional.log_softmax(x, dim=1)

        example = (torch.tensor([1, 2, 3, 2, 5]), torch.tensor([0, 3]))

        mod = torch.jit.script(NanoSparseNN())
        with torch.no_grad():
            with enable_profiling_mode():
                for _ in range(5):
                    mod(*example)

        g = torch.jit.last_executed_optimized_graph()
        self.run_pass('remove_mutation', g)
        self.run_pass('constant_propagation', g)
        g_str = str(torch.jit.last_executed_optimized_graph())
        # import pdb; pdb.set_trace()
        print(g_str)

        class SimpleMLP(torch.nn.Module):
            def __init__(self):
                super(SimpleMLP, self).__init__()
                self.fc1 = nn.Linear(100, 10)
                self.fc2 = nn.Linear(10, 1)

            def forward(self, x):
                x = nn.functional.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        example = (torch.randn(100),)
        mod = torch.jit.script(SimpleMLP())
        self.run_pass('remove_mutation', mod.forward.graph)
        self.run_pass('inline', mod.forward.graph)
        self.run_pass('peephole', mod.forward.graph)

        # import pdb; pdb.set_trace()

        with torch.no_grad():
            with enable_profiling_mode():
                for _ in range(5):
                    mod(*example)

        g = torch.jit.last_executed_optimized_graph()
        self.run_pass('remove_mutation', g)
        self.run_pass('constant_propagation', g)
        g_str = str(torch.jit.last_executed_optimized_graph())
        # import pdb; pdb.set_trace()
        print(g_str)



def get_constant(x):
    if x == inf:
        return 'float(\'inf\')' if PY2 else 'math.inf'
    if x == -inf:
        return 'float(\'-inf\')' if PY2 else '-math.inf'
    return x


def get_script_args(args):
    formals = []
    tensors = []
    actuals = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            name = 'i{}'.format(len(formals))
            formals.append(name)
            actuals.append(name)
            tensors.append(arg)
        elif isinstance(arg, str):
            actuals.append("'{}'".format(arg))
        else:
            actuals.append(str(get_constant(arg)))
    return (formals, tensors, actuals)


def get_call(method_name, func_type, args, kwargs):
    kwargs_str = ', '.join([k + '=' + str(v) for k, v in kwargs.items()])
    self_arg = args[0]
    if(func_type == 'method'):
        args = args[1:]

    argument_str = ', '.join(args)
    argument_str += ', ' if len(args) and len(kwargs) else ''
    argument_str += kwargs_str

    if func_type == 'functional':
        call = 'torch.{}({})'.format(method_name, argument_str)
    elif func_type == 'method':
        call = '{}.{}({})'.format(self_arg, method_name, argument_str)
    elif func_type == 'nn_functional':
        call = 'torch.nn.functional.{}({})'.format(method_name, argument_str)
    else:
        raise 'Unsupported function type'

    return call

# create a script function from (name, func_type, output_process_fn),
# returns a function takes in (args, kwargs) and runs the compiled function and
# then applies the post process fn to the outputs
def create_script_fn(self, method_name, func_type, output_process_fn):
    def script_fn(*args, **kwargs):
        formals, tensors, actuals = get_script_args(args)
        call = get_call(method_name, func_type, actuals, kwargs)
        script = script_template.format(', '.join(formals), call)

        CU = torch.jit.CompilationUnit(script)
        self.assertExportImport(CU.the_method.graph, tensors)
        output = output_process_fn(CU.the_method(*tensors))
        script_fn.last_graph = CU.the_method.graph_for(*tensors)
        return output
    return script_fn


def saved_last_module():
    pass

def check_against_reference(self, func, reference_func, args, kwargs=None,
                            allow_unused=True, check_types=True, no_grad=False):
    kwargs = kwargs if kwargs else {}

    saved_last_module.last_module = None

    def allSum(vs):
        if isinstance(vs, torch.Tensor):
            vs = (vs,)
        return sum((i + 1) * v.sum()
                   for i, v in enumerate(vs)
                   if v is not None and v.dtype.is_floating_point)

    def clone_inputs(requires_grad):
        inputs = [
            arg.detach().clone().requires_grad_(requires_grad and arg.requires_grad)
            if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        return inputs, [input for input in inputs if isinstance(input, torch.Tensor) and input.requires_grad]

    nograd_inputs, nograd_tensors = clone_inputs(False)
    recording_inputs, recording_tensors = clone_inputs(True)

    # test no gradients case
    outputs = self.runAndSaveRNG(reference_func, nograd_inputs, kwargs)
    with enable_profiling_mode():
        outputs_test = self.runAndSaveRNG(func, nograd_inputs, kwargs)


    self.assertEqual(outputs, outputs_test)
    with enable_profiling_mode():
        for _ in range(5):
            outputs_test = self.runAndSaveRNG(func, nograd_inputs, kwargs)

    def num_non_tensor_nodes(block):
        for node in block.nodes():
            kind = node.kind()
            if kind == "prim::Constant" or "prim::Bailout" in kind or "GetAttr" in kind:
                continue
            tensor_input = 0
            # for inp in node.inputs():
            #     if "Tensor" in str(inp.type()):
            #         tensor_input += 1
            #         break
            tensor_out = False
            for out in node.outputs():
                if "Tensor" in str(out.type()):
                    tensor_out = True
            if not tensor_out:
                # import pdb; pdb.set_trace()
                print(block)


    g = torch.jit.last_executed_optimized_graph()
    g_str = str(torch.jit.last_executed_optimized_graph())
    original_graph = g_str
    g_str = g_str[0:g_str.find("return")]
    count = g_str.count("prim::If") + g_str.count("prim::Loop") + g_str.count("aten::append") \
        + g_str.count("_set_item") + g_str.count("_(")
    if count > 0:
        original_count = count
        g_str2 = g_str
        self.run_pass('remove_mutation', g)
        self.run_pass('constant_propagation', g)
        g_str = str(torch.jit.last_executed_optimized_graph())
        g_str = g_str[0:g_str.find("return")]
        count = g_str.count("prim::If") + g_str.count("prim::Loop") + g_str.count("aten::append") \
            + g_str.count("_set_item") + g_str.count("_(")
        num_non_tensor_nodes(g)
        if (count != original_count):
            print("\n ORIGINAL COUNT", original_count, " NEW COUNT ", count, "\n")
            # import pdb; pdb.set_trace()
            # num_non_tensor_nodes(g.block())
            # if count > 0 and g_str.count("batch_norm") == 0:
            #     print(torch.jit.last_executed_optimized_graph())

def write_to_file(g, file_name):
    g_str = str(g)
    text_file = open("file_name", "w")
    text_file.write(g_str)



def suppress_warnings(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True):
            return fn(*args, **kwargs)
    return wrapper

class TestJitFunctional(JitTestCase):
    pass

script_template = '''
def the_method({}):
    return {}
'''

script_method_template = '''
def forward({}):
    return {}
'''

def add_nn_module_test(*args, **kwargs):
    if 'module_name' in kwargs:
        name = kwargs['module_name']
    elif 'fullname' in kwargs:
        name = kwargs['fullname']
    elif 'constructor' in kwargs:
        name = kwargs['constructor'].__name__

    no_grad = False if 'no_grad' not in kwargs else kwargs['no_grad']

    module_name = name.split("_")[0]

    if 'desc' in kwargs and 'eval' in kwargs['desc']:
        # eval() is not supported, so skip these tests
        return

    test_name = name
    if 'desc' in kwargs:
        test_name = "{}_{}".format(test_name, kwargs['desc'])
    test_name = 'test_nn_{}'.format(test_name)

    @suppress_warnings
    def do_test(self):
        if 'constructor' in kwargs:
            nn_module = kwargs['constructor']
        else:
            nn_module = getattr(torch.nn, name)

        if "FunctionalModule" in str(nn_module):
            return

        if 'constructor_args_fn' in kwargs:
            constructor_args = kwargs['constructor_args_fn']()
        else:
            constructor_args = kwargs.get('constructor_args', ())

        # Construct a script module that passes arguments through
        # to self.submodule
        def create_script_module(*args, **kwargs):
            formals, tensors, actuals = get_script_args(args)

            method_args = ', '.join(['self'] + actuals)
            call_args_str = ', '.join(actuals)
            call = "self.submodule({})".format(call_args_str)
            script = script_method_template.format(method_args, call)

            submodule_constants = []
            if kwargs.get('is_constant'):
                submodule_constants = ['submodule']

            # Create module to use the script method
            class TheModule(torch.jit.ScriptModule):
                __constants__ = submodule_constants

                def __init__(self):
                    super(TheModule, self).__init__()
                    self.submodule = nn_module(*constructor_args)

            def make_module(script):
                module = TheModule()
                # check __repr__
                str(module)
                module.define(script)
                return module

            # module cannot be imported / exported
            with torch.jit._disable_emit_hooks():
                if saved_last_module.last_module is None:
                    module = make_module(script)
                    saved_last_module.last_module = module
                else:
                    module = saved_last_module.last_module
                create_script_module.last_graph = module.graph
                mod = module(*args)
            return mod

        # Construct a normal nn module to stay consistent with create_script_module
        # and make use of a single global rng_state in module initialization
        def create_nn_module(*args, **kwargs):
            module = nn_module(*constructor_args)
            return module(*args)

        # Set up inputs from tuple of sizes or constructor fn
        if 'input_fn' in kwargs:
            input = kwargs['input_fn']()
        else:
            input = (kwargs['input_size'],)

        # Extra parameters to forward()
        if 'extra_args' in kwargs:
            input = input + kwargs['extra_args']

        if 'target_size' in kwargs:
            input = input + (kwargs['target_size'],)
        elif 'target_fn' in kwargs:
            if torch.is_tensor(input):
                input = (input,)
            input = input + (kwargs['target_fn'](),)

        args_variable, kwargs_variable = create_input(input)
        f_args_variable = deepcopy(unpack_variables(args_variable))
        print(test_name)

        # Check against Python module as reference
        check_against_reference(self, create_script_module, create_nn_module, f_args_variable, no_grad=no_grad)

    # if "conv" in test_name.lower():
    post_add_test(test_name, (), do_test, TestJitFunctional)


def post_add_test(test_name, skipTestIf, do_test, test_class):
    assert not hasattr(test_class, test_name), 'Two tests have the same name: ' + test_name

    for skip in skipTestIf:
        do_test = skip(do_test)

    setattr(test_class, test_name, do_test)


for test in module_tests + new_module_tests:
    add_nn_module_test(**test)
