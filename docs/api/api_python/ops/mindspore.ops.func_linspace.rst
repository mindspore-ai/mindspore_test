mindspore.ops.linspace
======================

.. py:function:: mindspore.ops.linspace(start, end, steps)

    创建一个steps个元素的，在[start, end]区间内均匀分布的一维tensor。

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [start, start+step, start+2*step, ... , end]
        \end{aligned}

    参数：
        - **start** (Union[Tensor, int, float]) - 区间的起始值。
        - **end** (Union[Tensor, int, float]) - 区间的末尾值。
        - **steps** (Union[Tensor, int]) - 元素数量。

    返回：
        Tensor
