mindspore.Tensor.log1p
======================

.. py:method:: mindspore.Tensor.log1p()

    对输入Tensor逐元素加一后计算自然对数。

    .. math::
        out_i = \log_e(x_i + 1)

    .. note::
        输入Tensor。上述公式中的 :math:`x` 。其值必须大于-1。

    返回：
        Tensor，与 `self` 的shape相同。

    异常：
        - **TypeError** - `self` 不是Tensor。
