mindspore.numpy.radians
=======================

.. py:function:: mindspore.numpy.radians(x, dtype=None)

    将角从角度制转换为弧度制。

    参数：
        - **x** (Tensor) - 角度制角。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor，对应的弧度制角。 如果 `x` 是Tensor标量，则返回Tensor标量。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。