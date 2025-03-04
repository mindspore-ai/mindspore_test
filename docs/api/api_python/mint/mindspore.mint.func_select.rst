mindspore.mint.select
=======================

.. py:function:: mindspore.mint.select(input, dim, index)

    在给定索引处沿选定维度对输入Tensor进行切片。

    .. warning::
        这是一个实验性API，可能会更改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dim** (int) - 进行切片的维度。
        - **index** (int) - 要选择的索引。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。