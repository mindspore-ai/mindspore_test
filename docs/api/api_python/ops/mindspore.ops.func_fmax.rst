mindspore.ops.fmax
==================

.. py:function:: mindspore.ops.fmax(input, other)

    逐元素计算输入tensor的最大值。

    .. math::
        output_i = \max(x1_i, x2_i)

    .. note::
        - 支持隐式类型转换、类型提升。
        - 输入 `input` 和 `other` 的shape必须能相互广播。
        - 如果其中一个比较值是NaN，则返回另一个比较值。

    参数：
        - **input** (Tensor) - 第一个输入。
        - **other** (Tensor) - 第二个输入。

    返回：
        Tensor
