mindspore.nn.HSigmoid
=============================

.. py:class:: mindspore.nn.HSigmoid

    逐元素计算Hard Sigmoid激活函数。

    Hard Sigmoid定义为：

    .. math::
        \text{HSigmoid}(input) =
        \begin{cases}
        0, & \text{ if } input \leq -3, \\
        1, & \text{ if } input \geq +3, \\
        input/6 + 1/2, & \text{ otherwise }
        \end{cases}

    HSigmoid函数图：

    .. image:: ../images/HSigmoid.png
        :align: center

    输入：
        - **input** (Tensor) - Hard Sigmoid的输入。

    输出：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 不是int或者float类型。
