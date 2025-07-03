mindspore.ops.igamma
====================

.. py:function:: mindspore.ops.igamma(input, other)

    计算正则化的下层不完全伽马函数。

    如果我们将 `input` 比作 `a` ， `other` 比作 `x` ，则正则化的下层不完全伽马函数可以表示成：

    .. math::
        P(a, x) = Gamma(a, x) / Gamma(a) = 1 - Q(a, x)

    其中，

    .. math::
        Gamma(a, x) = \int_0^x t^{a-1} \exp^{-t} dt

    为下层不完全伽马函数。

    :math:`Q(a, x)` 则为正则化的上层完全伽马函数。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **other** (Tensor) - 第二个输入tensor。

    返回：
        Tensor
