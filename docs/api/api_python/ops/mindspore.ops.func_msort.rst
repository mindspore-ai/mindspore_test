mindspore.ops.msort
====================

.. py:function:: mindspore.ops.msort(input)

    将输入Tensor的元素沿其第一个维度按值升序排序。

    `ops.msort(t)` 等价于 `ops.Sort(axis=0)(t)[0]`。更多信息请参考 :class:`mindspore.ops.Sort()`。

    .. Note::
        当前Ascend后端只支持对一维输入进行排序。

    参数：
        - **input** (Tensor) - 需要排序的输入，类型必须是float16或float32。

    返回：
        排序后的Tensor，与输入的shape和dtype一致。

    异常：
        - **TypeError** - `input` 的类型不是float16或float32。
