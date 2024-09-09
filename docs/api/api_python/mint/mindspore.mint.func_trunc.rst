mindspore.mint.trunc
===========================

.. py:function:: mindspore.mint.trunc(input)

    返回一个新的Tensor，该Tensor具有输入元素的截断整数值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入任意维度的Tensor。

    返回：
        Tensor， shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是 Tensor。