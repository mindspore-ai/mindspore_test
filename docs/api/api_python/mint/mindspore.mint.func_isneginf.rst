mindspore.mint.isneginf
=======================

.. py:function:: mindspore.mint.isneginf(input)

    确定输入Tensor每个位置上的元素是否为负无穷。

    .. warning::
        - 该API目前只支持在Atlas A2训练系列产品上使用。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，与输入具有相同形状，其中元素在对应输入为负无穷大时为 ``True``，否则为 ``False``。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。