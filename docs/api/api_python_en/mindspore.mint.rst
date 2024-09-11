mindspore.mint
===============

mindpsore.mint provides a large number of functional, nn, optimizer interfaces. The API usages and functions are consistent with the mainstream usage in the industry for easy reference.
The mint interface is currently an experimental interface and performs better than ops in graph mode of O0 and PyNative mode. Currently, the graph sinking mode and CPU/GPU backend are not supported, and it will be gradually improved in the future.

The module import method is as follows:

.. code-block::

    from mindspore import mint

Tensor
---------------

Creation Operations
^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.arange
    mindspore.mint.eye
    mindspore.mint.full
    mindspore.mint.linspace
    mindspore.mint.ones
    mindspore.mint.ones_like
    mindspore.mint.zeros
    mindspore.mint.zeros_like

Indexing, Slicing, Joining, Mutating Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.cat
    mindspore.mint.gather
    mindspore.mint.index_select
    mindspore.mint.masked_select
    mindspore.mint.permute
    mindspore.mint.scatter_add
    mindspore.mint.split
    mindspore.mint.narrow
    mindspore.mint.nonzero
    mindspore.mint.tile
    mindspore.mint.stack
    mindspore.mint.where

Random Sampling
-----------------

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.normal
    mindspore.mint.rand_like
    mindspore.mint.rand

Math Operations
------------------

Pointwise Operations
^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.abs
    mindspore.mint.add
    mindspore.mint.acos
    mindspore.mint.acosh
    mindspore.mint.arccos
    mindspore.mint.arccosh
    mindspore.mint.arcsin
    mindspore.mint.arcsinh
    mindspore.mint.arctan
    mindspore.mint.arctan2
    mindspore.mint.arctanh
    mindspore.mint.asin
    mindspore.mint.asinh
    mindspore.mint.atan
    mindspore.mint.atan2
    mindspore.mint.atanh
    mindspore.mint.bitwise_and
    mindspore.mint.bitwise_or
    mindspore.mint.bitwise_xor
    mindspore.mint.ceil
    mindspore.mint.clamp
    mindspore.mint.cos
    mindspore.mint.cosh
    mindspore.mint.cross
    mindspore.mint.div
    mindspore.mint.divide
    mindspore.mint.erf
    mindspore.mint.erfc
    mindspore.mint.erfinv
    mindspore.mint.exp
    mindspore.mint.expm1
    mindspore.mint.fix
    mindspore.mint.floor
    mindspore.mint.log
    mindspore.mint.log1p
    mindspore.mint.logical_and
    mindspore.mint.logical_not
    mindspore.mint.logical_or
    mindspore.mint.logical_xor
    mindspore.mint.mul
    mindspore.mint.neg
    mindspore.mint.negative
    mindspore.mint.pow
    mindspore.mint.reciprocal
    mindspore.mint.remainder
    mindspore.mint.roll
    mindspore.mint.round
    mindspore.mint.rsqrt
    mindspore.mint.sigmoid
    mindspore.mint.sin
    mindspore.mint.sinc
    mindspore.mint.sinh
    mindspore.mint.sqrt
    mindspore.mint.square
    mindspore.mint.sub
    mindspore.mint.tan
    mindspore.mint.tanh
    mindspore.mint.xlogy
    mindspore.mint.trunc

Reduction Operations
^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.argmax
    mindspore.mint.argmin
    mindspore.mint.all
    mindspore.mint.any
    mindspore.mint.max
    mindspore.mint.mean
    mindspore.mint.median
    mindspore.mint.min
    mindspore.mint.prod
    mindspore.mint.sum
    mindspore.mint.unique

Comparison Operations
^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.eq
    mindspore.mint.greater
    mindspore.mint.greater_equal
    mindspore.mint.gt
    mindspore.mint.isclose
    mindspore.mint.isfinite
    mindspore.mint.le
    mindspore.mint.less
    mindspore.mint.less_equal
    mindspore.mint.lt
    mindspore.mint.maximum
    mindspore.mint.minimum
    mindspore.mint.ne
    mindspore.mint.topk
    mindspore.mint.sort

BLAS and LAPACK Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.baddbmm
    mindspore.mint.bmm
    mindspore.mint.inverse
    mindspore.mint.matmul
    mindspore.mint.trace

Other Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.broadcast_to
    mindspore.mint.cummax
    mindspore.mint.cummin
    mindspore.mint.cumsum
    mindspore.mint.flatten
    mindspore.mint.flip
    mindspore.mint.repeat_interleave
    mindspore.mint.searchsorted

mindspore.mint.nn
------------------

Loss Functions
^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.L1Loss

Convolution Layers
^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Fold
    mindspore.mint.nn.Unfold

Normalization Layers
^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.GroupNorm

Non-linear Activations (weighted sum, nonlinearity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Hardshrink
    mindspore.mint.nn.Hardsigmoid
    mindspore.mint.nn.Hardswish
    mindspore.mint.nn.LogSoftmax
    mindspore.mint.nn.Mish
    mindspore.mint.nn.ReLU
    mindspore.mint.nn.SELU
    mindspore.mint.nn.Softshrink

Linear Layers
^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Linear

Dropout Layers
^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Dropout

Pooling Layers
^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.AvgPool2d

Loss Functions
^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.BCEWithLogitsLoss

mindspore.mint.nn.functional
-----------------------------

Convolution functions
^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.fold
    mindspore.mint.nn.functional.unfold

Pooling functions
^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.avg_pool2d
    mindspore.mint.nn.functional.max_pool2d

Non-linear activation functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.batch_norm
    mindspore.mint.nn.functional.elu
    mindspore.mint.nn.functional.gelu
    mindspore.mint.nn.functional.group_norm
    mindspore.mint.nn.functional.hardshrink
    mindspore.mint.nn.functional.hardsigmoid
    mindspore.mint.nn.functional.hardswish
    mindspore.mint.nn.functional.layer_norm
    mindspore.mint.nn.functional.leaky_relu
    mindspore.mint.nn.functional.log_softmax
    mindspore.mint.nn.functional.mish
    mindspore.mint.nn.functional.relu
    mindspore.mint.nn.functional.selu
    mindspore.mint.nn.functional.sigmoid
    mindspore.mint.nn.functional.silu
    mindspore.mint.nn.functional.softmax
    mindspore.mint.nn.functional.softplus
    mindspore.mint.nn.functional.softshrink
    mindspore.mint.nn.functional.tanh

Linear functions
^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.linear

Dropout functions
^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.dropout

Sparse functions
^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.embedding
    mindspore.mint.nn.functional.one_hot

Loss Functions
^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.binary_cross_entropy
    mindspore.mint.nn.functional.binary_cross_entropy_with_logits
    mindspore.mint.nn.functional.l1_loss

Vision functions
^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.grid_sample
    mindspore.mint.nn.functional.pad

mindspore.mint.optim
---------------------

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.optim.AdamW

mindspore.mint.linalg
----------------------

Inverses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.linalg.inv

mindspore.mint.special
----------------------

Pointwise Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.special.erfc
    mindspore.mint.special.expm1
    mindspore.mint.special.log1p
    mindspore.mint.special.log_softmax
    mindspore.mint.special.round
    mindspore.mint.special.sinc
