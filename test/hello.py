import torch
import weakref

from _weakref import (
     getweakrefcount,
     getweakrefs,
     ref,
     proxy,
     CallableProxyType,
     ProxyType,
     ReferenceType,
     _remove_dead_weakref)

from _weakrefset import WeakSet, _IterationGuard

import _collections_abc  # Import after _weakref to avoid circular import.
import sys
import itertools
import contextlib

@contextlib.contextmanager
def monkeypatched(object, name, patch):
    """ Temporarily monkeypatches an object. """

    pre_patched_value = getattr(object, name)
    setattr(object, name, patch)
    yield object
    setattr(object, name, pre_patched_value)

def id_eq(self, other):
    import pdb; pdb.set_trace()
    return True

# class TensorIdDict(dict):
#     def __getitem__(self, key):
#         with monkeypatched(torch.Tensor, "__eq__", id_eq):
#             out = super().__getitem__(key)
#             return out

#     def __setitem__(self, key, value):
#         with monkeypatched(torch.Tensor, "__eq__", id_eq):
            # return super().__setitem__(key, value)
            
    # def __delitem__(self, key):
        # with monkeypatched(torch.Tensor, "__eq__", id_eq):
        #     return super().__delitem__(key)

# class WeakTensorKeyDictionary(weakref.WeakKeyDictionary):
#     def __init__(self):
#         super(WeakTensorKeyDictionary, self).__init__()
#         self.data = TensorIdDict()

class WeakTensorRefHolder(object):
    def __init__(self, val):
        self.val = weakref.ref(val)
    
    def __hash__(self):
        return id(self.val())

    def __eq__(self, other):
        if self.val() is None or other.val() is None:
            return False
        return self.val() is other.val()

x = {}

ten = torch.tensor([1, 2])
abc = WeakTensorRefHolder(ten)
x[abc] = 10
print(WeakTensorRefHolder(ten) in x)

def del_ten():
    del x[abc]

weakref.finalize(ten, del_ten)
del ten
print(x)
print('hello')


# x = torch.Tensor([2,3,4])
# dic = WeakTensorKeyDictionary()
# dic[x] = 5
# y = torch.Tensor([1, 2])
# dic[y] = 15
# tensors = []
# for i in range(1000000):
#     ten = torch.tensor([1, i])
#     tensors.append(ten)
#     dic[ten] = 2
# # print(dic[x])
# # dic[x] = 10
# # print(dic[x])
# # del x
# print(list(dic.keys()))
# print(d[x])
