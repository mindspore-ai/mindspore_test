mindspore.ops.truncate_div
==========================

.. py:function:: mindspore.ops.truncate_div(x, y)

    将 `x` 和 `y` 逐元素相除，结果将向0取整。

    .. note::
        支持隐式类型转换，支持广播。

    参数：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入tensor。

    返回：
        Tensor
