mindspore.mint.take
===================

.. py:function:: mindspore.mint.take(input, index)

    选取给定索引 `index` 处的 `input` 元素。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入张量。
        - **index** (LongTensor) - 输入张量的索引张量。

    返回：
        Tensor，shape与索引的shape相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `index` 的数据类型不是long。