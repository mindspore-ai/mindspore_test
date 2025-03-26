mindspore.ops.unique_consecutive
================================

.. py:function:: mindspore.ops.unique_consecutive(input, return_inverse=False, return_counts=False, dim=None)

    对输入tensor中每组连续且重复的元素去重，仅保留首个元素。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **return_inverse** (bool, 可选) - 是否返回输入tensor中各元素映射到输出中的位置。默认 ``False`` 。
        - **return_counts** (bool, 可选) - 是否返回每个唯一值出现的次数。默认 ``False`` 。
        - **dim** (int, 可选) - 指定去重的维度。默认 ``None`` ，表示对输入进行展平。

    返回：
        Tensor或由多个tensor组成的tuple(output, inverse_indices, counts)。

        - **output** (Tensor) - 去重后的输出。
        - **inverse_indices** (Tensor, optional) - 输入tensor的元素在 `output` 中的索引。
        - **counts** (Tensor, optional) - 每个唯一值出现的次数。
