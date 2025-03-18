mindspore.ops.bessel_y1
=======================

.. py:function:: mindspore.ops.bessel_y1(x)

    逐元素计算输入tensor的第二类一阶贝塞尔函数值。

    .. math::
        \begin{array}{ll} \\
            Y_{1}(x)=\lim_{n \to 1} \frac{J_{n}(x) \cos n \pi-J_{-n}(x)}{\sin n \pi}
        \end{array}

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor
