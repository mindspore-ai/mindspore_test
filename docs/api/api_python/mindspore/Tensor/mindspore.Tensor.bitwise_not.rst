mindspore.Tensor.bitwise_not
=============================

.. py:method:: mindspore.Tensor.bitwise_not() -> Tensor

    逐元素对当前Tensor进行位取反。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    返回：
        Tensor，其数据类型和shape与 `self` 相同。

    异常：
        - **TypeError** - 如果 `self` 不是Tensor。
        - **RuntimeError** - 如果 `self` 的数据类型不是int或bool。
