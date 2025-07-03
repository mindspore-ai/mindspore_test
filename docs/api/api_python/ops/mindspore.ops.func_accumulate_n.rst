mindspore.ops.accumulate_n
==========================

.. py:function:: mindspore.ops.accumulate_n(x)

    逐元素计算列表中各个tensor的和。

    :func:`mindspore.ops.accumulate_n` 与 :func:`mindspore.ops.addn` 类似，但accumulate_n不会等待其所有输入就绪后再求和，可节省内存。

    参数：
        - **x** (Union(tuple[Tensor], list[Tensor])) - 输入tensors。

    返回：
        Tensor

