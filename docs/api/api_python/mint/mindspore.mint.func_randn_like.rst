mindspore.mint.randn_like
=========================

.. py:function:: mindspore.mint.randn_like(input, *, dtype=None)

    返回shape与输入相同，类型为 `dtype` 的Tensor。dtype由输入决定，其元素取值服从 :math:`[0, 1)` 区间内的正态分布。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入的Tensor。用来决定输出Tensor的shape和默认的dtype。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定输出Tensor的dtype，必须是float类型。如果是 ``None`` ，则使用输入Tensor的dtype。默认值： ``None`` 。

    返回：
        Tensor，shape和dtype由输入决定其元素为服从标准正态分布的数字。
