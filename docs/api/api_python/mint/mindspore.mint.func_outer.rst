mindspore.mint.outer
====================

.. py:function:: mindspore.mint.outer(input, vec2)

    计算 `input` 和 `vec2` 的外积。如果向量 `input` 长度为 :math:`n` ， `vec2` 长度为 :math:`m` ，则输出矩阵shape为 :math:`(n, m)` 。

    .. note::
        该函数不支持广播。

    参数：
        - **input** (Tensor) - 输入一维向量。
        - **vec2** (Tensor) - 输入一维向量。

    返回：
        out，两个一维向量的外积，是一个二维矩阵。

    异常：
        - **TypeError** - 如果 `input` 或 `vec2` 不是Tensor。
        - **TypeError** - `input` 和 `vec2` 隐式转换后的数据类型不是float16、float32、float64、bool、uint8、int8、int16、int32、int64、complex64、complex128、bfloat16之一
        - **ValueError** - 如果 `input` 或 `vec2` 的维度不是1。