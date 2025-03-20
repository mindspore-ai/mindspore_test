mindspore.mint.argwhere
========================

.. py:function:: mindspore.mint.argwhere(input)

    返回一个Tensor，包含所有输入Tensor非零数值的位置。

    .. warning::
        这是一个实验性API，后续可能修改或删除

    参数：
        - **input** (Tensor) - 输入Tensor。类型可以为Number或bool。

    返回：
        一个2-D Tensor，数据类型为int64，包含所有输入中的非零数值的位置。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **ValueError** - 如果 `input` 的维度等于0。
