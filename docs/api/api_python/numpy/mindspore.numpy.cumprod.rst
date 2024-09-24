mindspore.numpy.cumprod
=======================

.. py:function:: mindspore.numpy.cumprod(a, axis=None, dtype=None)

    返回沿给定 `axis` 的元素的累计乘积。

    .. note::
        不支持NumPy参数 `out` 。

    参数：
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 输入Tensor。
        - **axis** (int, 可选) - 计算累积乘积所沿的轴。 默认情况下输入会被展开。 默认值： ``None`` 。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入不能转换为Tensor或者 `axis` 不是整数。
        - **ValueError** - 如果 `axis` 超出范围。