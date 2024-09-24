mindspore.numpy.rad2deg
=======================

.. py:function:: mindspore.numpy.rad2deg(x, dtype=None)

    将角从弧度制转换为角度制。

    参数：
        - **x** (Tensor) - 弧度制角。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。 覆盖输出Tensor的dtype。

    返回：
        Tensor，对应的角度制角。 如果 `x` 是Tensor标量，则返回Tensor标量。