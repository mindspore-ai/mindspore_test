mindspore.ops.lerp
==================

.. py:function:: mindspore.ops.lerp(input, end, weight)

    基于权重参数计算两个tensor之间的线性插值。

    .. math::

        output_{i} = input_{i} + weight_{i} * (end_{i} - input_{i})

    .. note::
        - 输入 `input` 和 `end` 的shape必须是可广播的。
        - 如果 `weight` 为tensor，则 `weight` 、 `input` 和 `end` 的shape必须是可广播的。
        - 在Ascend平台上，若 `weight` 为浮点数，则 `input` 与 `end` 类型应为float32。

    参数：
        - **input** (Tensor) - 起始点。
        - **end** (Tensor) - 终止点。
        - **weight** (Union[float, Tensor]) - 线性插值公式的权重参数。

    返回：
        Tensor
