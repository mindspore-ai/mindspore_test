mindspore.ops.hypot
====================

.. py:function:: mindspore.ops.hypot(input, other)

    给定直角三角形的边，逐元素计算其斜边。

    支持广播、类型提升。

    .. math::
        out_i = \sqrt{input_i^2 + other_i^2}

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **other** (Tensor) - 第二个输入tensor。

    返回：
        Tensor
