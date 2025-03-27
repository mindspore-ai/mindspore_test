mindspore.ops.bessel_j0
=======================

.. py:function:: mindspore.ops.bessel_j0(x)

    逐元素计算输入tensor的第一类零阶贝塞尔函数值。

    .. math::
        \begin{array}{ll} \\
            J_{0}(x) = \frac{1}{\pi} \int_{0}^{\pi} \cos (x \sin \theta) d \theta
            =\sum_{m=0}^{\infty} \frac{(-1)^{m} x^{2 m}}{2^{2 m} (m !)^2}
        \end{array}

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor
