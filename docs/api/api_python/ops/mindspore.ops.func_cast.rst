mindspore.ops.cast
===================

.. py:function:: mindspore.ops.cast(input, dtype)

    转换输入Tensor的数据类型。

    .. note::
        将复数转换为bool类型的时候，不考虑复数的虚部，只要实部不为零，返回True，否则返回False。

    参数：
        - **input** (Union[Tensor, Number]) - 输入要进行数据类型转换的Tensor，其shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dtype** (dtype.Number) - 指定转换的数据类型。仅支持常量值。

    返回：
        Tensor，其shape与 `input` 相同，即 :math:`(x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - `input` 既不是Tensor也不是数值型。
        - **TypeError** - `dtype` 不是数值型。
