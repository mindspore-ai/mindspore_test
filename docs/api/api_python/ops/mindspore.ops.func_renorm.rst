mindspore.ops.renorm
====================

.. py:function:: mindspore.ops.renorm(input, p, axis, maxnorm)

    返回一个tensor，其中按指定轴计算每个子tensor的 `p` 范数小于等于 `maxnorm`。如果大于，返回子tensor上的原始值除以子tensor的 `p` 范数，然后再乘以 `maxnorm` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **p** (int) - 范数计算的幂。
        - **axis** (int) - 指定计算轴。
        - **maxnorm** (float32) - 指定的最大范数。

    返回：
        Tensor
