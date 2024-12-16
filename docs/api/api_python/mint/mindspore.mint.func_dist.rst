mindspore.mint.dist
===================

.. py:function:: mindspore.mint.dist(input, other, p=2)

    计算两个行向量集合中每对之间的批处理 :math:`p`-范数距离。

    参数：
        - **input** (Tensor) - 第一个输入张量。数据类型必须是 float16 或 float32。
        - **other** (Tensor) - 第二个输入张量。数据类型必须是 float16 或 float32。
        - **p** (float, 可选) - 范数的阶数。 `p` 必须大于或等于 0，默认值为 ``2``。

    返回：
        返回一个与 `input` 数据类型相同的张量，其shape为 :math:`(1)`。

    异常：
        - **TypeError** - 如果 `input` 或 `other` 不是Tensor。
        - **TypeError** - 如果 `input` 或 `other` 的数据类型既不是 float16 也不是 float32。
        - **TypeError** - 如果 `p` 不是非负实数。
