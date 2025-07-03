mindspore.ops.cummin
====================

.. py:function:: mindspore.ops.cummin(input, axis)

    返回tensor在指定轴上的累积最小值及其索引。

    .. math::
        \begin{array}{ll} \\
            y_{i} = \min(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 指定计算的轴。

    返回：
        两个tensor组成的tuple(min, min_indices)
