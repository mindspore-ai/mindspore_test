mindspore.nn.SoftShrink
========================

.. py:class:: mindspore.nn.SoftShrink(lambd=0.5)

    逐元素计算SoftShrink激活函数。

    .. math::
        \text{SoftShrink}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    SoftShrink函数图：

    .. image:: ../images/Softshrink.png
        :align: center

    参数：
        - **lambd** (number, 可选) - SoftShrink公式中的 :math:`\lambda` ，必须不小于零。默认值： ``0.5`` 。

    输入：
        - **input** (Tensor) - SoftShrink的输入，任意维度的Tensor，数据类型为float16或float32。

          - Ascend: float16、float32、bfloat16。
          - CPU/GPU: float16、float32。
    输出：
        Tensor，shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - `lambd` 不是float、int或bool。
        - **TypeError** - `input` 不是tensor。
        - **TypeError** - `input` 的数据类型float16、float32或bfloat16。