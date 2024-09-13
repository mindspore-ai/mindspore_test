mindspore.numpy.ogrid
=================================

.. py:function:: mindspore.numpy.ogrid()

    返回一个稀疏矩阵 ``NdGrid`` 实例，其中 ``sparse=True`` 。
    输出数组的维度和数量等于索引维度的数量。如果 ``step`` 不是一个复数，则 ``stop`` 不包括在内。然而，如果 ``step`` 是复数（例如5j），那么它的整数部分被解释为指定要在 ``start`` 和 ``stop`` 之间创建的点的数量，其中 ``stop`` 包括在内。

    .. note::
        在graph模式下不受支持。与Numpy不同，如果 ``step`` 是一个带有实数分量的复数，则 ``step`` 被处理为等效于 ``int(abs(step))`` 。

    异常：
        - **TypeError** - 如果切片索引不是整数。