mindspore.ops.unique
====================

.. py:function:: mindspore.ops.unique(input)

    对输入tensor中元素去重。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        由两个tensor组成的tuple(output, indices)。

        - **output** (Tensor) - 去重后的输出。
        - **indices** (Tensor) - 输入tensor的元素在 `output` 中的索引。
