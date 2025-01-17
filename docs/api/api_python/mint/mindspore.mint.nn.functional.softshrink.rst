mindspore.mint.nn.functional.softshrink
=======================================

.. py:function:: mindspore.mint.nn.functional.softshrink(input, lambd=0.5)

    Soft Shrink激活函数。按输入元素计算输出。公式定义如下：

    .. math::
        \text{SoftShrink}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    SoftShrink激活函数图：

    .. image:: ../images/Softshrink.png
        :align: center

    参数：
        - **input** (Tensor) - Soft Shrink的输入。支持数据类型：

          - Ascend：float16、float32、bfloat16。
        - **lambd** (number，可选) - Soft Shrink公式定义的阈值 :math:`\lambda` 。 `lambd` 应该大于等于0。默认值： ``0.5`` 。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `lambd` 不是float、bool或int。
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的dtype不是float16，float32或bfloat16。