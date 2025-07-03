mindspore.ops.logit
===================

.. py:function:: mindspore.ops.logit(input, eps=None)

    逐元素计算tensor的逻辑回归函数值。

    公式如下：

    .. math::
        \begin{align}
        y_{i} & = \ln(\frac{z_{i}}{1 - z_{i}}) \\
        z_{i} & = \begin{cases}
        input_{i} & \text{if eps is None} \\
        \text{eps} & \text{if } input_{i} \lt \text{eps} \\
        input_{i} & \text{if } \text{eps} \leq input_{i} \leq 1 - \text{eps} \\
        1 - \text{eps} & \text{if } input_{i} \gt 1 - \text{eps}
        \end{cases}
        \end{align}

    参数：
        - **input** (Tensor) - 输入tensor。
        - **eps** (float, 可选) - 用于输入限制边界的epsilon值。当eps不是None时，输入的数值界限被定义[eps, 1-eps]，否则没有界限。
          默认 ``None`` 。

    返回：
        Tensor
