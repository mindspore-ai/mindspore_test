mindspore.mint.triu
===================

.. py:function:: mindspore.mint.triu(input, diagonal=0)

    将指定对角线下方的元素设置为0。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **diagonal** (int，可选) - 二维tensor的指定对角线。默认 ``0`` ，表示主对角线。

    返回：
        Tensor