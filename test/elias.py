import torch
import itertools
from torch.testing._internal.jit_utils import RUN_CUDA
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    FakeTensorConverter,
    DynamicOutputShapeException,
)
from torch.utils._python_dispatch import enable_torch_dispatch_mode
from torch import nn
import unittest
import contextlib
import torch._prims as prims
import copy
from torch import Tensor


def meta_embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):    
    assert weight.dim() == 2, "'weight' must be 2-D"
    # TODO: Assert not ported over yet
    #   auto indices_arg = TensorArg(indices, "indices", 1);
    #   checkScalarTypes("embedding", indices_arg, {kLong, kInt});

    if indices.dim() == 1:
        return weight.index_select(0, indices)

    size = list(indices.shape)
    for d in weight.shape[1:]:
        size.append(d)

    embedding =  weight.index_select(0, indices.reshape(-1))

    # only used for mean
    eq_padding_count = None

    if padding_idx != -1:
        eq_padding = indices == padding_idx
        if mode == "sum":
            scalar_val = 0.
        elif mode == "max":
            scalar_val = -inf
        elif mode == "mean":
            scalar_val = 0.
            eq_padding_count = eq_padding.sum(dim=1)
        embedding = torch.where(eq_padding.reshape(-1).unsqueeze(-1), embedding.new_full([1], scalar_val), embedding)
    
    if per_sample_weights is not Noen:
        embedding = embedding * per_sample_weight

    embedding = embedding.view(size)
    if mode == "sum":
        return torch.sum(embedding, dim=1)
    elif mode == "mean":
        # TOOD: compute sum higher precision
        temp = torch.sum(embedding, dim=1)
        div = (indices.new_full([1], embedding.size(1)) - eq_padding_count)
        return temp / div.unsqueeze(-1)
    else:
        assert mode == "max"
        return torch.max(embedding, dim=1)

# batch, sequence, features = dims(3)
# r = embedding_weights[input[batch, sequence], features].sum(sequence)
# r.order(batch, features)

mode_str = "mean"
embedding = nn.EmbeddingBag(6, 3, mode=mode_str, padding_idx=1)
 
# out[0] 
embedding.weight = torch.nn.Parameter(torch.arange(0., 18.).reshape([6, 3]))
input = torch.LongTensor([[1,2,4,5],[4,3,2,0]])
out1 = embedding(input)
out2 = embedding_ref(embedding.weight, input, mode=mode_str, padding_idx=1)

print(out1, "\n\n\n\n", out2)
torch.testing.assert_close(out1, out2)