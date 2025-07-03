mindspore.mint.nn.functional.hardswish
=======================================

.. py:function:: mindspore.mint.nn.functional.hardswish(input)

    Hard Swish激活函数。输入是一个Tensor，具有任何有效的shape。

    Hard Swish定义如下：

    .. math::
        \text{HardSwish}(input) =
        \begin{cases}
        0, & \text{ if } input \leq -3, \\
        input, & \text{ if } input \geq +3, \\
        input*(input + 3)/6, & \text{ otherwise }
        \end{cases}

    HardSwish函数图：

    .. image:: ../images/Hardswish.png
        :align: center

    参数：
        - **input** (Tensor) - Hard Swish的输入。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `input` 不是一个Tensor。
        - **TypeError** - `input` 不是int或者float类型。