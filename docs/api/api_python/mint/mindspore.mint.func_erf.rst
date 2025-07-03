mindspore.mint.erf
==================

.. py:function:: mindspore.mint.erf(input)

    逐元素计算输入tensor的高斯误差。

    .. math::

        \text{erf}(x)=\frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
