mindspore.Tensor.isneginf
=========================

.. py:method:: mindspore.Tensor.isneginf()

    逐元素判断是否是负inf。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 对于Ascend，仅支持 Atlas A2 训练系列产品。

    返回：
        Tensor，shape与输入shape相同，对应 `self` 元素为负inf的位置是 ``true`` ，反之为 ``false`` 。
