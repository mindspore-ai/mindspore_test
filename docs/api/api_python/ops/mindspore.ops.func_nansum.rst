mindspore.ops.nansum
====================

.. py:function:: mindspore.ops.nansum(input, axis=None, keepdims=False, *, dtype=None)

    忽略NaN值，按指定轴计算输入tensor的和。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **axis** (Union[int, tuple(int)], 可选) - 求和的维度。假设 `input` 的秩为r，取值范围[-r,r)，默认 ``None`` ，对tensor中的所有元素求和。
        - **keepdims** (bool, 可选) - 输出tensor是否保持维度，默认 ``False`` ，不保留维度。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出tensor的类型，默认 ``None`` 。

    返回：
        Tensor
