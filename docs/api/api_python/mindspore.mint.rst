mindspore.mint
===============

mindspore.mint提供了大量的functional、nn、优化器接口，API用法及功能等与业界主流用法一致，方便用户参考使用。
mint接口当前是实验性接口，在图编译模式为O0和PyNative模式下性能比ops更优。当前暂不支持图下沉模式及CPU、GPU后端，后续会逐步完善。

模块导入方法如下：

.. code-block::

    from mindspore import mint

MindSpore中 `mindspore.mint` 接口与上一版本相比，新增、删除和支持平台的变化信息请参考 `mindspore.mint API接口变更 <https://gitee.com/mindspore/docs/blob/master/resource/api_updates/mint_api_updates_cn.md>`_ 。

Tensor
---------------

创建运算
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
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

索引、切分、连接、突变运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.cat
    mindspore.mint.gather
    mindspore.mint.index_select
    mindspore.mint.masked_select
    mindspore.mint.permute
    mindspore.mint.scatter
    mindspore.mint.scatter_add
    mindspore.mint.split
    mindspore.mint.narrow
    mindspore.mint.nonzero
    mindspore.mint.tile
    mindspore.mint.tril
    mindspore.mint.stack
    mindspore.mint.where

随机采样
------------

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.multinomial
    mindspore.mint.normal
    mindspore.mint.rand_like
    mindspore.mint.rand

数学运算
------------------

逐元素运算
^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
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
    mindspore.mint.exp2
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
    mindspore.mint.mv
    mindspore.mint.nan_to_num
    mindspore.mint.neg
    mindspore.mint.negative
    mindspore.mint.pow
    mindspore.mint.reciprocal
    mindspore.mint.remainder
    mindspore.mint.roll
    mindspore.mint.round
    mindspore.mint.rsqrt
    mindspore.mint.sigmoid
    mindspore.mint.sign
    mindspore.mint.sin
    mindspore.mint.sinc
    mindspore.mint.sinh
    mindspore.mint.softmax
    mindspore.mint.sqrt
    mindspore.mint.square
    mindspore.mint.sub
    mindspore.mint.tan
    mindspore.mint.tanh
    mindspore.mint.trunc
    mindspore.mint.xlogy

Reduction运算
^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
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

比较运算
^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
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
    mindspore.mint.not_equal
    mindspore.mint.topk
    mindspore.mint.sort

BLAS和LAPACK运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.baddbmm
    mindspore.mint.bmm
    mindspore.mint.inverse
    mindspore.mint.matmul
    mindspore.mint.trace

其他运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
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
    mindspore.mint.tril

mindspore.mint.nn
------------------

损失函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.L1Loss

卷积层
^^^^^^^^^^^^^^^^^^
.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Fold
    mindspore.mint.nn.Unfold

归一化层
^^^^^^^^^^^^^^^^^^
.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.GroupNorm

非线性激活层 (加权和，非线性)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.GELU
    mindspore.mint.nn.Hardshrink
    mindspore.mint.nn.Hardsigmoid
    mindspore.mint.nn.Hardswish
    mindspore.mint.nn.LogSoftmax
    mindspore.mint.nn.Mish
    mindspore.mint.nn.PReLU
    mindspore.mint.nn.ReLU
    mindspore.mint.nn.SELU
    mindspore.mint.nn.Softmax
    mindspore.mint.nn.Softshrink

线性层
^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Linear

Dropout层
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Dropout

池化层
^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.AvgPool2d

损失函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.BCEWithLogitsLoss
    mindspore.mint.nn.MSELoss

mindspore.mint.nn.functional
-----------------------------

卷积函数
^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.fold
    mindspore.mint.nn.functional.unfold

池化函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.avg_pool2d
    mindspore.mint.nn.functional.max_pool2d

非线性激活函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
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
    mindspore.mint.nn.functional.prelu
    mindspore.mint.nn.functional.relu
    mindspore.mint.nn.functional.selu
    mindspore.mint.nn.functional.sigmoid
    mindspore.mint.nn.functional.silu
    mindspore.mint.nn.functional.softmax
    mindspore.mint.nn.functional.softplus
    mindspore.mint.nn.functional.softshrink
    mindspore.mint.nn.functional.tanh

线性函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.linear

Dropout函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.dropout

稀疏函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.embedding
    mindspore.mint.nn.functional.one_hot

损失函数
^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.binary_cross_entropy
    mindspore.mint.nn.functional.binary_cross_entropy_with_logits
    mindspore.mint.nn.functional.l1_loss
    mindspore.mint.nn.functional.mse_loss

Vision函数
^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.grid_sample
    mindspore.mint.nn.functional.pad

mindspore.mint.optim
---------------------

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.optim.AdamW

mindspore.mint.linalg
----------------------

逆数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.linalg.inv

mindspore.mint.special
----------------------

逐元素运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.special.erfc
    mindspore.mint.special.exp2
    mindspore.mint.special.expm1
    mindspore.mint.special.log1p
    mindspore.mint.special.log_softmax
    mindspore.mint.special.round
    mindspore.mint.special.sinc

mindspore.mint.distributed
--------------------------------

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.distributed.all_gather
    mindspore.mint.distributed.all_gather_into_tensor
    mindspore.mint.distributed.all_gather_object
    mindspore.mint.distributed.all_reduce
    mindspore.mint.distributed.all_to_all
    mindspore.mint.distributed.all_to_all_single
    mindspore.mint.distributed.barrier
    mindspore.mint.distributed.batch_isend_irecv
    mindspore.mint.distributed.broadcast
    mindspore.mint.distributed.broadcast_object_list
    mindspore.mint.distributed.destroy_process_group
    mindspore.mint.distributed.gather
    mindspore.mint.distributed.gather_object
    mindspore.mint.distributed.get_backend
    mindspore.mint.distributed.get_global_rank
    mindspore.mint.distributed.get_group_rank
    mindspore.mint.distributed.get_process_group_ranks
    mindspore.mint.distributed.get_rank
    mindspore.mint.distributed.get_world_size
    mindspore.mint.distributed.init_process_group
    mindspore.mint.distributed.irecv
    mindspore.mint.distributed.isend
    mindspore.mint.distributed.new_group
    mindspore.mint.distributed.P2POp
    mindspore.mint.distributed.recv
    mindspore.mint.distributed.reduce
    mindspore.mint.distributed.reduce_scatter
    mindspore.mint.distributed.reduce_scatter_tensor
    mindspore.mint.distributed.scatter
    mindspore.mint.distributed.scatter_object_list
    mindspore.mint.distributed.send
