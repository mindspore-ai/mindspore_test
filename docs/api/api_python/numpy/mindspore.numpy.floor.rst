mindspore.numpy.floor
=====================

.. py:function:: mindspore.numpy.floor(x, dtype=None)

    返回输入的向下取整，逐元素计算。

    标量 `x` 的向下取整是满足 ``i <= x`` 的最大整数 `i` 。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。
        在GPU上，支持的数据类型有np.float16和np.float32。 在CPU上，支持的数据类型有np.float16，np.float32和np.float64。

    参数：
        - **x** (Tensor) - 输入数据。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。 `x` 中各元素的向下取整。 如果 `x` 是标量，返回标量。