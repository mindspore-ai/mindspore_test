mindspore.numpy.swapaxes
=================================

.. py:function:: mindspore.numpy.swapaxes(x, axis1, axis2)

    交换Tensor的两个轴。

    参数：
        - **x** (Tensor) - 要调换的Tensor。
        - **axis1** (int) - 第一个轴。
        - **axis2** (int) - 第二个轴。

    返回：
        调换后的Tensor，其数据类型与原始Tensor ``x`` 相同。

    异常：
        - **TypeError** - 如果 ``axis1`` 或 ``axis2`` 不是整数，或 ``x`` 不是Tensor。
        - **ValueError** - 如果 ``axis1`` 或 ``axis2`` 不在 :math:`[-ndim, ndim-1]` 的范围内。