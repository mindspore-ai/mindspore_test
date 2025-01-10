mindspore.Tensor.outer
======================

.. py:method:: mindspore.Tensor.outer(vec2)

    计算 `self` 和 `vec2` 的外积。如果向量 `self` 长度为 :math:`n` ， `vec2` 长度为 :math:`m` ，则输出矩阵shape为 :math:`(n, m)` 。

    .. note::
        该函数不支持广播。

    参数：
        - **vec2** (Tensor) - 输入一维向量。

    返回：
        out (Tensor, 可选)，两个一维向量的外积，是一个二维矩阵。

    异常：
        - **TypeError** - 如果 `vec2` 不是Tensor。