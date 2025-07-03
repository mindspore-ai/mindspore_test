mindspore.ops.random_gamma
==========================

.. py:function:: mindspore.ops.random_gamma(shape, alpha, seed=None)

    根据伽马分布生成随机数。

    参数：
        - **shape** (Tensor) - 指定生成随机数的shape。
        - **alpha** (Tensor) - :math:`\alpha` 分布的参数。必须是非负数。
        - **seed** (int, 可选) - 随机种子，必须为非负数。默认 ``None`` 。

    返回：
        Tensor。形状为 `mindspore.ops.concat([shape, alpha.shape], axis=0)` 。数据类型和alpha一致。
