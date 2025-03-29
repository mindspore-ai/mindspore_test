mindspore.ops.logspace
======================

.. py:function:: mindspore.ops.logspace(start, end, steps, base=10, *, dtype=mstype.float32)

    返回一个 `steps` 个元素的，在 [ :math:`base^{start}` , :math:`base^{end}` ] 区间内均匀分布的一维tensor。

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [base^{start}, base^{start + 1 * step}, ... , base^{start + (steps-2) * step}, base^{end}]
        \end{aligned}

    参数：
        - **start** (Union[float, Tensor]) - 区间的起始值。
        - **end** (Union[float, Tensor]) - 区间的末尾值。
        - **steps** (int) - 元素数量。
        - **base** (int，可选) - 对数函数的底数。默认 ``10`` 。

    关键字参数：
        - **dtype** (mindspore.dtype，可选) - 指定数据类型。默认 ``mstype.float32`` 。

    返回：
        Tensor

