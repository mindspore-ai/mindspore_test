mindspore.ops.amax
==================

.. py:function:: mindspore.ops.amax(input, axis=None, keepdims=False, *, initial=None, where=None)

    返回tensor在指定轴上的最大值。

    参数：
        - **input** (Tensor[Number]) - 输入tensor。
        - **axis** (Union[int, tuple(int), list(int), Tensor], 可选) - 指定计算的轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None`` 。
        - **keepdims** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    关键字参数：
        - **initial** (scalar, 可选) - 最大值的初始值。默认 ``None`` 。
        - **where** (Tensor[bool], 可选) - 指定计算最大值的范围，该tensor的shape须可被广播到 `input` 的shape上。必须指定initial值。默认 ``None`` ，表示计算全部元素。

    返回：
        Tensor
