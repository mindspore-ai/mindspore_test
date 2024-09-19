mindspore.numpy.ravel
=================================

.. py:function:: mindspore.numpy.ravel(x)

    返回一个连续的展平Tensor。
    返回一个一维Tensor，包含输入Tensor的所有元素。

    参数：
        - **x** (Tensor) - 需要展平的Tensor。

    返回：
        展平后的Tensor，与原始Tensor ``x`` 具有相同的数据类型。

    异常：
        - **TypeError** - 如果 ``x`` 不是Tensor。