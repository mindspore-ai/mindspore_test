mindspore.Tensor.expm1
======================

.. py:method:: mindspore.Tensor.expm1()

    逐元素计算 `self` 的指数，然后减去1。

    .. math::
        out_i = e^{x_i} - 1

    .. note::
        指数函数的输入Tensor。上述公式中的 :math:`x` 。

    返回：
        Tensor，shape与 `self` 相同。

    异常：
        - **TypeError** - 如果 `self` 不是Tensor。
