mindspore.mint.take_along_dim
=============================

.. py:function:: mindspore.mint.take_along_dim(input, indices, dim=None)

    从 `input` 中根据 `indices` 沿指定的 `dim` 维度选择值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入张量。
        - **indices** (Tensor) - 输入张量的索引张量, 一定要是long类型。
        - **dim** (int, 可选) - 选择的维度。
    
    返回：
        Tensor，元素来自于 `input` 。
