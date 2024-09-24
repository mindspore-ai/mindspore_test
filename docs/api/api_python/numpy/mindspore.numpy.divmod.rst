mindspore.numpy.divmod
======================

.. py:function:: mindspore.numpy.divmod(x1, x2, dtype=None)

    同时返回逐元素的商和余数。

    参数：
        - **x1** (Union[Tensor]) - 被除Tensor。
        - **x2** (Union[Tensor, int, float, bool]) - 除数。如果 ``x1.shape != x2.shape`` ，它们必须能被广播到一个共同的形状。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        逐元素返回商和向下取整除的余数，格式为(商，余数)

    异常：
        - **TypeError** - 如果 `x1` 和 `x2` 不是Tensor或Tensor。