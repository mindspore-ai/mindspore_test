mindspore.mint.nn.functional.hardsigmoid
=========================================

.. py:function:: mindspore.mint.nn.functional.hardsigmoid(input)

    Hard Sigmoid激活函数。按元素计算输出。

    Hard Sigmoid定义为：

    .. math::
        \text{HardSigmoid}(input) =
        \begin{cases}
        0, & \text{ if } input \leq -3, \\
        1, & \text{ if } input \geq +3, \\
        input/6 + 1/2, & \text{ otherwise }
        \end{cases}

    HardSigmoid函数图：

    .. image:: ../images/Hardsigmoid.png
        :align: center

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 不是int或者float类型。
