mindspore.Tensor.exp
=====================

.. py:method:: Tensor.exp()

    逐元素计算 `self` 的指数。

    .. math::

        out_i = e^{x_i}

    .. note::
        指数函数的输入Tensor。上述公式中的 :math:`x` 。

    返回：
        Tensor，具有与 `self` 相同的shape。

    异常：
        - **TypeError** - `self` 不是Tensor。
