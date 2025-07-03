mindspore.ops.dot
==================

.. py:function:: mindspore.ops.dot(input, other)

    计算两个输入tensor的点积。

    .. note::
        - 输入为float16或float32，且秩必须大于或等于2。

    参数：
        - **input** (Tensor) - 第一个输入的tensor。
        - **other** (Tensor) - 第二个输入的tensor。

    返回：
        Tensor