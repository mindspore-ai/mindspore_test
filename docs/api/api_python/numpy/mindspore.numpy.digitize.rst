mindspore.numpy.digitize
========================

.. py:function:: mindspore.numpy.digitize(x, bins, right=False)

    返回输入数组中每个值所属的桶的索引。如果 `x` 中的值超出了桶的界限，将返回 0 或 ``len(bins)`` 作为相应的索引。

    参数：
       - **x** (Union[int, float, bool, list, tuple, Tensor]) - 待分桶的输入数组。
       - **bins** (Union[list, tuple, Tensor]) - 桶的数组。必须是一维且单调的。
       - **right** (boolean, 可选) - 表示区间是否包含右边界或左边界。默认为 ``(right==False)`` ，表示区间不包括右边界。在这种情况下，左边界是开放的，即 ``bins[i-1] <= x < bins[i]`` 是单调递增桶的默认行为。

    返回：
        元素为int的Tensor，输出元素为索引的数组，与 `x` 有相同的shape。