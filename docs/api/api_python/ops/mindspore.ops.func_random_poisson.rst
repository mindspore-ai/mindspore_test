mindspore.ops.random_poisson
============================

.. py:function:: mindspore.ops.random_poisson(shape, rate, seed=None, dtype=mstype.float32)

    从指定均值为 `rate` 的泊松分布中，生成形状为 `shape` 的随机样本。

    .. math::

        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}
    
    .. warning::
        Ascend后端不支持随机数重现功能， `seed` 参数不起作用。

    参数：
        - **shape** (Tensor) - 表示要从每个分布中采样的随机数张量的形状。一维整型tensor。
        - **rate** (Tensor) - 泊松分布的 :math:`μ` 参数，表示泊松分布的均值，同时也是分布的方差。
        - **seed** (int, 可选) - 随机数种子。必须是一个非负整数，默认 ``None`` 。
        - **dtype** (mindspore.dtype) - 返回的数据类型。默认 ``mstype.float32`` 。

    返回：
        Tensor, 形状为 `mindspore.ops.concat([shape, rate.shape], axis=0)` 。
