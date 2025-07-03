mindspore.ops.roll
===================

.. py:function:: mindspore.ops.roll(input, shifts, dims=None)

    按维度移动tensor的元素。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **shifts** (Union[list(int), tuple(int), int]) - 元素移动量。
        - **dims** (Union[list(int), tuple(int), int], 可选) - 指定移动维度，默认 ``None`` ，表示将输入tensor展平后再进行计算，然后将计算结果reshape为输入的shape。

    返回：
        Tensor