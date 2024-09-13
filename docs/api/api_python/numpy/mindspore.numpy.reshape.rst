mindspore.numpy.reshape
=================================

.. py:function:: mindspore.numpy.reshape(x, new_shape)

    在不改变数据的情况下重塑一个Tensor。

    参数：
        - **x** (Tensor) - 需要重塑的Tensor。
        - **new_shape** (Union[int, list(int), tuple(int)]) - Tensor的新shape。必须与原始shape兼容。如果 ``new_shape`` 是一个只有一个元素的tuple，则结果将是一个具有该长度的一维Tensor。 ``new_shape`` 中的一个维度可以是-1，此时该维度的值将根据Tensor的总长度和其他维度的大小推断得出。

    返回：
        重塑后的Tensor，其数据类型与原始Tensor ``x`` 相同。

    异常：
        - **TypeError** - 如果 ``new_shape`` 既不是整数、list或tuple，或者 ``x`` 不是Tensor。
        - **ValueError** - 如果 ``new_shape`` 与 ``x`` 的原始shape不兼容。