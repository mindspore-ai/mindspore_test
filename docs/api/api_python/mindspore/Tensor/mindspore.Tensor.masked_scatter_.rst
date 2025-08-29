mindspore.Tensor.masked_scatter\_
=================================

.. py:method:: mindspore.Tensor.masked_scatter_(mask, source) -> Tensor

    根据 `mask` ，使用 `source` 中的值，更新 `self` 的值，返回一个Tensor。 `mask` 和 `self` 的shape必须相等或者 `mask` 是可广播的。

    .. note::
        当 `source` 中的元素总数少于 `mask` 中True元素的个数时，NPU可能无法拦截该非法输入，因此无法保证输出结果的正确性。

    参数：
        - **mask** (Tensor[bool]) - 一个bool Tensor，其shape可以被广播到 `self` 。
        - **source** (Tensor) - 一个Tensor，其数据类型与 `self` 相同。 `source` 中的元素数量必须大于等于 `mask` 中的True元素的数量。

    返回：
        Tensor，其数据类型和shape与 `self` 相同。

    异常：
        - **TypeError** - 如果 `mask` 或者 `source` 不是Tensor。
        - **TypeError** - 如果Tensor本身的数据类型不被支持。
        - **TypeError** - 如果 `mask` 的dtype不是bool。
        - **TypeError** - 如果Tensor本身的维度数小于 `mask` 的维度数。
        - **ValueError** - 如果 `mask` 不能广播到Tensor本身。
