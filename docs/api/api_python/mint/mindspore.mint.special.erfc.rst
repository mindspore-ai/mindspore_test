mindspore.mint.special.erfc
============================

.. py:function:: mindspore.mint.special.erfc(input)

    逐元素计算输入tensor的互补误差。

    .. math::

        \text{erfc}(x) = 1 - \frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
