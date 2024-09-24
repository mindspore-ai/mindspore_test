mindspore.numpy.power
=====================

.. py:function:: mindspore.numpy.power(x1, x2, dtype=None)

    第一个数组的元素以第二个数组的元素为指数逐元素求幂。

    将 `x1` 中的每个基数以 `x2` 中对应位置的元素为指数取幂。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。在GPU上，支持的数据类型有np.float16和np.float32。

    参数：
        - **x1** (Tensor) - 基数。
        - **x2** (Tensor) - 指数。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量， `x1` 中的基数按 `x2` 中的指数计算幂的结果。如果 `x1` 和 `x2` 都是标量，返回标量。
    