mindspore.mint.trace
====================

.. py:function:: mindspore.mint.trace(input)

    返回 `input` 的主对角线方向上的总和。

    参数：
        - **input** (Tensor) - 二维Tensor。

    返回：
        Tensor，当 `input` 为数据类型为整型或bool时其数据类型为int64，反之与 `input` 一致，含有一个元素。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **ValueError** - 如果 `input` 的维度不是2。
        - **TypeError** - `input` 的数据类型不是float16、float32、float64、bool、uint8、int8、int16、int32、int64、complex64、complex128、bfloat16之一。