mindspore.mint.count_nonzero
============================

.. py:function:: mindspore.mint.count_nonzero(input, dim=None)

    计算输入Tensor指定轴上的非零元素的数量。如果没有指定维度，则计算Tensor中所有非零元素的数量。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 要计算非零元素个数的输入。其shape为 :math:`(*)` ，其中 :math:`*` 为任意维度。
        - **dim** (Union[None, int, tuple(int), list(int)], 可选) - 要沿其计算非零值数量的维度。默认值： ``None`` ，计算所有非零元素的个数。求和的维度。如果 `dim` 为 ``None`` ，对Tensor中的所有元素求和。

    返回：
        Tensor，指定的轴上非零元素数量。

    异常：
        - **TypeError** - `input` 不是Tensor类型。
        - **TypeError** - `dim` 类型不是int，tuple(int)，list(int)或None。
        - **ValueError** - `dim` 取值不在 :math:`[-input.ndim, input.ndim)` 范围。
