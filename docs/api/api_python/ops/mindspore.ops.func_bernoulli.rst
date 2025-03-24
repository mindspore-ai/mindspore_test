mindspore.ops.bernoulli
=======================

.. py:function:: mindspore.ops.bernoulli(input, p=0.5, seed=None)

    根据伯努利分布生成随机数（0或1）。

    .. math::

        out_{i} \sim Bernoulli(p_{i})

    参数：
        - **input** (Tensor) - 输入tensor。
        - **p** (Union[Tensor, float], 可选) - 输出tensor中对应位置为1的概率。数值范围在0到1之间。默认 ``0.5`` 。
        - **seed** (Union[int, None], 可选) - 随机种子。默认 ``None`` 表示使用时间戳。

    返回：
        Tensor