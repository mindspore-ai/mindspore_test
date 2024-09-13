mindspore.numpy.broadcast_arrays
=================================

.. py:function:: mindspore.numpy.broadcast_arrays(*args)

    将任意数量的数组广播到共同的shape。

    .. note::
        不支持Numpy的 ``subok`` 参数。在图模式下，返回的不是Tensor列表，而是Tensor的tuple。

    参数：
        - **\*args** (Tensor) - 要进行广播的数组。

    返回：
        Tensor列表。

    异常：
        - **ValueError** - 如果数组不能被广播。