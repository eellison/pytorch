
# need to get correctness conditions 
# if forward - backward has been called
# if just inference, refcount to 0 ?

# set up the tape guarantees

# also need to set up liveness guarantees 
# the same graph outputs from a previous graph are live now as before
# delta of what was alive previously, and what died now


# dont copy outputs of a previous cudagraph 

# set up thread local, clean up pool 


# allow multiple different output tapes so long as predecessors are the smae
# This doesnt really matter for training, since the backward will be fixed 
# but you could re-record backward if needed
# but that might be slow..  



# Now I need to 

# set up thread local, when to clean up pool

# allow different execution traces so long as they are not concurrent
# this is essential to allowing different sizes


# we need to not join two tapes if all of the entries from the previous tape are dead
# 
# e.g. invoke resnet with one set of outputs
# invoke resnet with the other set of outputs 

# for inference, we might want to include some logic about copying final outputs
# avoid lifetime overlap

# TODO - for cudagraph inputs, we should be checking that they die before invoking, not
# after (note this only matters for cudagraph inputs). we can just delete them 
# from the inputs and then check that the corresponding weakref output has died


import torch
import torch._dynamo
x = torch.rand([4, 4], device="cuda")

@torch._dynamo.optimize()
def foo(x):
    return x * x * 10

data_ptr = foo(x).data_ptr()
out = foo(x)
assert out.data_ptr() == data_ptr
out2 = foo(x)
assert out.data_ptr() != out2.data_ptr()


# @torch._dynamo.optimize()
# def foo2(x, y):
#     return x

# out1, out2 = foo(x)
# del out2
# out3 = foo2(out1, torch.rand([4], device="cuda"))

# del out3
# out1, out2 = foo(x)
# out3 = foo2(out1, torch.rand([4], device="cuda"))

# inps = torch.rand([4, 4]).cuda()
# foo(inps).sum().backward()

# inps2 = torch.rand([10, 4]).cuda()
# foo(inps2).sum().backward()
