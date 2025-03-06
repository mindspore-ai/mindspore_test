mindspore.ops.atan2
===================

.. py:function:: mindspore.ops.atan2(input, other)

    逐元素计算input/other的反正切值。

    返回 :math:`\theta\ \in\ [-\pi, \pi]` ，使得 :math:`input = r*\sin(\theta), other = r*\cos(\theta)` ，
    其中 :math:`r = \sqrt{input^2 + other^2}` 。

    参数：
        - **input** (Tensor, Number.number) - 输入tensor或常数。
        - **other** (Tensor, Number.number) - 输入tensor或常数。

    返回：
        Tensor
