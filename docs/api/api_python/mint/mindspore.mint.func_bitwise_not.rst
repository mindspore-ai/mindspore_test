mindspore.mint.bitwise_not
=============================

.. py:function:: mindspore.mint.bitwise_not(input)

    逐元素对input进行位取反。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor，数据类型必须是int或bool。

    返回：
        Tensor，其数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **RuntimeError** - 如果 `input` 的数据类型不是int或bool。
