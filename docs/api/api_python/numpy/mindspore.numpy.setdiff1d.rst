mindspore.numpy.setdiff1d
=================================

.. py:function:: mindspore.numpy.setdiff1d(ar1, ar2, assume_unique=False)

    计算两个Tensor的差集。返回在 ``ar1`` 中，不在 ``ar2`` 中的唯一值。

    参数：
        - **ar1** (Union[int, float, bool, list, tuple, Tensor]) - 输入Tensor。
        - **ar2** (Union[int, float, bool, list, tuple, Tensor]) - 输入Tensor。
        - **assume_unique** (bool，可选) - 如果为 ``True`` ，则假定输入Tensor没有重复元素，这可以加速计算。如果为 ``True`` 但 ``ar1`` 或 ``ar2`` 有重复元素时，可能会导致不正确的结果或超出范围的索引。默认值: ``False`` 。

    返回：
        一维Tensor，包含在 ``ar1`` 中但不在 ``ar2`` 中的值。当 ``assume_unique`` 为 ``False`` 时，结果会被排序，否则仅当输入已排序时才排序。

    异常：
        - **TypeError** - 如果输入的 ``ar1`` 或 ``ar2`` 不是类似数组的对象。
        - **TypeError** - 如果 ``assume_unique`` 不是bool类型。