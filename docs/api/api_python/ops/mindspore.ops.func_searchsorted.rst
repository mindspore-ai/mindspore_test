mindspore.ops.searchsorted
==========================

.. py:function:: mindspore.ops.searchsorted(sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None)

    返回元素能够插入输入tensor的位置索引，以维持原tensor的递增顺序。

    参数：
        - **sorted_sequence** (Tensor) - 输入tensor。如果未提供 `sorter` ，最内层的维度上须为递增的序列。
        - **values** (Tensor) - 要插入元素的值。

    关键字参数：
        - **out_int32** (bool, 可选) - 输出数据类型是否为mindspore.int32。如果为 ``False`` ，则输出数据类型将为mindspore.int64。默认 ``False`` 。
        - **right** (bool, 可选) - 搜索策略。如果为 ``True`` ，则返回找到的最后一个合适的索引；如果为 ``False`` ，则返回第一个合适的索引。默认 ``False`` 。
        - **side** (str, 可选) - 跟参数 `right` 功能一致，如果参数值为 ``left``，相当于 `right` 为 ``False``。如果参数值为 ``right`` ，相当于 `right` 为 ``True``。如果值为 ``left`` 但是 `right` 为 ``True`` 则报错。默认 ``None`` 。
        - **sorter** (Tensor, 可选) - 按 `sorted_sequence` 最内层维度升序排序的索引序列，与未排序的 `sorted_sequence` 共同使用。CPU和GPU只支持 ``None`` 。默认 ``None`` 。

    返回：
        Tensor