mindspore.Tensor.hardshrink
===========================

.. py:method:: mindspore.Tensor.hardshrink(lambd=0.5)

    Hard Shrink激活函数。按输入元素计算输出。公式定义如下：

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    HardShrink激活函数图：

    .. image:: ../../images/Hardshrink.png
        :align: center

    .. note::
        指数函数的输入Tensor。上述公式中的 :math:`x` 。支持数据类型：

        - Ascend：float16、float32、bfloat16。
        - CPU/GPU：float16、float32。

    参数：
        - **lambd** (number，可选) - Hard Shrink公式定义的阈值 :math:`\lambda` 。默认值： ``0.5`` 。

    返回：
        Tensor，shape和数据类型与输入 `self` 相同。

    异常：
        - **TypeError** - `lambd` 不是float、int或bool。
        - **TypeError** - `self` 不是Tensor。
        - **TypeError** - `self` 的dtype不是float16、float32或bfloat16。
