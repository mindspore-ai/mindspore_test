mindspore.numpy.float_power
===========================

.. py:function:: mindspore.numpy.float_power(x1, x2, dtype=None)

    第一个数组逐元素计算幂次方，指数为第二个数组中对应的元素。

    将 `x1` 中的每个基数以其位置对应 `x2` 中的元素作为指数，计算幂次方。
    `x1` 和 `x2` 必须能广播到相同的shape。
    这与power函数的不同之处在于int、float16和float64都会提升为至少具有float32精度的浮点数，这样结果总是不精确的。
    这个函数意在为负幂返回一个可用的值，并且让正幂很少溢出。

    .. note::
        不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。
        int和float将被提升至float32而不是float64。

    参数：
        - **x1** (Tensor) - 基数。
        - **x2** (Tensor) - 指数。
        - **dtype** (mindspore.dtype，可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。 `x1` 中的基数以 `x2` 中的对应元素为指数计算得到的幂次方。 如果 `x1` 和 `x2` 都是标量，返回标量。