Fake Tensor
===========

.. currentmodule:: torch._subclasses

Fake tensors are tensors that have all the metadata of a real tensor (shape, dtype, device, strides)
but no actual data. They are useful for:

- **Shape inference**: Determining output shapes without running actual computation
- **Compilation**: Running compiler passes that need tensor metadata without memory allocation
- **Debugging**: Testing tensor operation behavior without using GPU memory

For a detailed guide on using FakeTensor with torch.compile, see the
`FakeTensor user guide <torch.compiler_fake_tensor.html>`_.

API Reference
-------------

FakeTensorMode
~~~~~~~~~~~~~~

.. autoclass:: FakeTensorMode
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __enter__, __exit__

    A context manager that intercepts tensor operations and converts them to work with fake tensors.

    Example::

        from torch._subclasses import FakeTensorMode

        # Create a fake mode
        fake_mode = FakeTensorMode()

        # Create real tensors
        real_tensor = torch.randn(10, 20)

        # Convert to fake tensor
        fake_tensor = fake_mode.from_tensor(real_tensor)

        # Operations in fake mode produce fake tensors
        with fake_mode:
            result = fake_tensor @ fake_tensor.T
            # result is a FakeTensor with shape [10, 10]

FakeTensor
~~~~~~~~~~

.. autoclass:: FakeTensor
    :members:
    :undoc-members:
    :show-inheritance:

    A tensor subclass that contains metadata but no actual data.

    FakeTensors are typically created via :class:`FakeTensorMode` rather than
    directly instantiated.

Exceptions
~~~~~~~~~~

.. autoexception:: UnsupportedFakeTensorException
    :show-inheritance:

    Raised when an operation is not supported with fake tensors.

.. autoexception:: DynamicOutputShapeException
    :show-inheritance:

    Raised when an operation would produce a dynamically-shaped output
    that cannot be determined from input shapes alone.

Utilities
~~~~~~~~~

.. autofunction:: torch._subclasses.fake_tensor.unset_fake_temporarily

    Context manager to temporarily disable fake tensor mode.

    Example::

        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with FakeTensorMode() as fake_mode:
            # Inside fake mode
            with unset_fake_temporarily():
                # Temporarily outside fake mode - can do real computation
                real_result = torch.randn(10).sum()
            # Back inside fake mode

Related Topics
--------------

- :doc:`torch.compile documentation <torch.compiler_api>`
- `FakeTensor user guide <user_guide/torch_compiler/torch.compiler_fake_tensor.html>`_
