mindspore.ops.gamma
====================

.. py:function:: mindspore.ops.gamma(shape, alpha, beta, seed=None)

    根据伽马分布生成随机数。

    支持广播。

    .. warning::
        Ascend后端不支持随机数重现功能， `seed` 参数不起作用。

    参数：
        - **shape** (tuple) - 指定shape。
        - **alpha** (Tensor) - shape参数。
        - **beta** (Tensor) - 逆尺度参数。
        - **seed** (int，可选) - 随机种子，默认 ``None`` 。

    返回：
        Tensor