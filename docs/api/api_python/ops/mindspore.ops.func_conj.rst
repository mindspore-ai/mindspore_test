mindspore.ops.conj
===================

.. py:function:: mindspore.ops.conj(input)

    逐元素计算输入tensor的共轭。复数的形式必须是 `a + bj` ，其中a是实部，b是虚部。

    返回的共轭形式为 `a - bj` 。

    如果 `input` 是实数，则直接返回 `input` 。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
