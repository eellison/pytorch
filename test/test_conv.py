import torch
from torch import nn
import timeit
# torch.set_num_threads(1)
torch.set_default_dtype(torch.float)
from torch.nn import functional as F


shapes_file = open("conv_input_values.txt", "r")
from ast import literal_eval
shapes = [literal_eval(line) for line in shapes_file.readlines()]



# # run([4096],[4096,2048])

# # [[256,1,512],[256,1,2048],[2048,512]]


i = 0

def warmup(fn, iter=2):
    for _ in range(iter):
        fn()

torch.set_grad_enabled(False)
# # need to check dtype, omp num threads, and bias or no bias
use_bias = False

NITER = 20
print("Input Shape", "Weight Shape", "With Bias", "No MKLDNN", "With MKLDNN", "Speedup %", "Num Threads", "Default Dtype", sep='\t')
import time

results_file = open("results_file_speedup_conv2d_new.txt", "a")
results_file.write("\t".join(["Input Shape", "Weight Shape", "With Bias", "No MKLDNN", "With MKLDNN", "Speedup %", "Num Threads", "Default Dtype"]))
results_file.write("\n")
for set_number_threads_1 in [True]:
    if set_number_threads_1:
        torch.set_num_threads(1)
    for shape in shapes:
        i += 1
        input_tensor, weight, bias = shape[0:3]
        # not supported by mkldnn linear
        if len(input_tensor) != 4:
            continue

        weight_size = weight
        input_tensor_size = input_tensor
        weight = torch.rand(weight)
        bias = torch.rand(bias) if bias else None

        # some of the top shapes errors out, not sure why
        try:
            torch.nn.functional.conv2d(torch.rand(input_tensor_size), torch.rand(weight_size))
        except Exception as e:
            print("Bad shape inputs", shape[0:3])
            results_file.write("Bad shape inputs" + str(shape[0:3]) + "\n")
            continue

        class Mod(torch.nn.Module):
            def __init__(self, weight, bias):
                super(Mod, self).__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, input):
                return torch.nn.functional.conv2d(input, self.weight, self.bias)

        mod_no_mkldnn = torch.jit.freeze(torch.jit.script(Mod(weight, bias).eval()))

        def no_mkldnn():
            x = torch.rand(input_tensor)
            return mod_no_mkldnn(x)

        warmup(no_mkldnn)
        # Time no mkldnn execution
        torch.manual_seed(0)
        s = time.time()
        results = []
        for _ in range(NITER):
            results.append(no_mkldnn())
        e = time.time()
        elapsed_no_mkldnn_time = (e - s) / NITER

        torch.manual_seed(0)
        result1 = no_mkldnn()
        mod_mkldnn = torch.jit.freeze(torch.jit.script(Mod(weight, bias).eval()))
        torch._C._jit_pass_convert_frozen_conv_to_mkldnn(mod_mkldnn.graph)

        def with_mkldnn():
            x = torch.rand(input_tensor)
            return mod_mkldnn(x)

        warmup(with_mkldnn)
        # Time mkldnn execution
        torch.manual_seed(0)
        s = time.time()
        results2 = []
        for _ in range(NITER):
            out = with_mkldnn()
            results2.append(out)
        e = time.time()
        elapsed_mkldnn_time = (e - s) / NITER

        torch.manual_seed(0)
        for elem1, elem2 in zip(results, results2):
            assert torch.allclose(elem1, elem2)

        speedup_pct = (elapsed_no_mkldnn_time - elapsed_mkldnn_time) / elapsed_no_mkldnn_time * 100

        num_threads = torch.get_num_threads()
        default_dtype = torch.get_default_dtype()

        print(input_tensor_size, weight_size, use_bias, elapsed_no_mkldnn_time, elapsed_mkldnn_time, speedup_pct, num_threads, default_dtype, sep='\t')

        out = input_tensor_size, weight_size, use_bias, elapsed_no_mkldnn_time, elapsed_mkldnn_time, speedup_pct, num_threads, default_dtype
        out = [str(x) for x in out]
        # import gc
        # gc.collect()
        # print(psutil.cpu_percent(), psutil.virtual_memory().percent)
        results_file.write("\t".join(out))
        results_file.write("\n")


# torch.clear_mkl_bufs()

# print(psutil.cpu_percent(), psutil.virtual_memory().percent)

results_file.close()



# [[1,512,14,14],[1024,512,1,1],[1024],[],[],[],[]]
# x = torch.rand(256, 512)
# y = torch.rand(2048, 512)
# print(timeit.timeit(lambda: torch._C._nn.linear(x, y), number=10))

# # y = y.to_mkldnn()
# print(timeit.timeit(lambda: torch._C._nn.linear(x.to_mkldnn(), y.to_mkl).to_dense(), number=10))

# # input = torch.rand([1,512,14,14])
# weight = torch.rand([1024,512,1,1])
# bias = torch.rand([1024])



# F.conv2d(input, weight, bias, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
