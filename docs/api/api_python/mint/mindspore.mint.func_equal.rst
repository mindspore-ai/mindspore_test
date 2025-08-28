mindspore.mint.equal
=====================

.. py:function:: mindspore.mint.equal(input, other)

    比较两个输入是否相等。

    .. note::
        `input` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。

    参数：
        - **input** (Tensor) - 第一个输入。
        - **other** (Tensor) - 第二个输入。

    返回：
        bool。

    异常：
        - **TypeError** - 如果 `input` 或 `other` 不是Tensor。