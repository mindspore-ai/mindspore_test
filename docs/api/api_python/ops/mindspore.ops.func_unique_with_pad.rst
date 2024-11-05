mindspore.ops.unique_with_pad
=============================

.. py:function:: mindspore.ops.unique_with_pad(x, pad_num)

    对输入一维Tensor中元素去重，返回一维Tensor中的唯一元素（使用pad_num填充）和相对索引。

    基本操作与unique相同，但unique_with_pad多了pad操作。
    unique运算符对Tensor处理后所返回的元组（ `y` ， `idx` ）， `y` 与 `idx` 的shape通常会有差别。因此，为了解决上述情况，
    unique_with_pad操作符将用用户指定的 `pad_num` 填充 `y` ，使其具有与 `idx` 相同shape。

    .. warning::
        :func:`mindspore.ops.unique_with_pad` 从2.4版本开始已被弃用，并将在未来版本中被移除。
        请组合使用 :func:`mindspore.ops.unique` 和 :func:`mindspore.ops.pad` 实现同样的功能。

    参数：
        - **x** (Tensor) - 需要被去重的Tensor。必须是类型为int32或int64的一维向量。
        - **pad_num** (int) - 填充值。数据类型为int32或int64。

    返回：
        tuple (Tensor)，包含两个Tensor，分别是 `y` 、 `idx` 。

        - **y** (Tensor) - `y` 是与 `x` shape和数据类型相同的Tensor，包含 `x` 中去重后的元素，并用 `pad_num` 填充。
        - **idx** (Tensor) - `idx` 为索引Tensor，包含 `x` 中的元素在 `y` 中的索引，与 `x` 的shape相同。

    异常：
        - **TypeError** - `x` 的数据类型既不是int32也不是int64。
        - **ValueError** - `x` 不是一维Tensor。
