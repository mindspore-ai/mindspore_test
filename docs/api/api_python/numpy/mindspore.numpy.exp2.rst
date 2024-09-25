mindspore.numpy.exp2
====================

.. py:function:: mindspore.numpy.exp2(x, dtype=None)

    计算输入数组中所有值 `p` 的 ``2**p`` 。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。
        在GPU上，支持的数据类型为np.float16和np.float32。

    参数：
        - **x** (Tensor) - 输入值。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量，逐元素计算2的x次幂。