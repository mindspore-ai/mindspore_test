mindspore.ops.fill\_
======================

.. py:function:: mindspore.ops.fill_(input, value)

    用指定的值填充 `input` 。

    参数：
        - **input** (Tensor) - 用来填充的Tensor。
        - **value** (Union(Tensor, number.Number, bool)) - 用来填充 `input` 的值。

    返回：
        Tensor。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **RunTimeError** - `input` 或 `value` 的数据类型不支持。
        - **RunTimeError** - 当 `value` 是Tensor时，它应该是0-D Tensor或shape=[1]的1-D Tensor。
