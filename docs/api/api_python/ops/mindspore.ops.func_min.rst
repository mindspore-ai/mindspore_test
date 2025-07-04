mindspore.ops.min
==============================

.. py:function:: mindspore.ops.min(input, axis=None, keepdims=False, *, initial=None, where=None)

    返回tensor在指定轴上的最小值及其索引。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 指定轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None`` 。
        - **keepdims** (bool) - 输出tensor是否保留维度。默认 ``False`` 。

    关键字参数：
        - **initial** (scalar, 可选) - 最小值的初始值。默认 ``None`` 。
        - **where** (Tensor[bool], 可选) - 指定计算最小值的范围，该tensor的shape须可被广播到 `input` 的shape上。必须指定initial值。默认 ``None`` ，表示计算全部元素。

    返回：
        两个tensor组成的tuple(min, min_indices)。