mindspore.Tensor.clone
======================

.. py:method:: mindspore.Tensor.clone() -> Tensor

    返回一个当前Tensor的副本。

    .. note::
        该函数是可微分的，梯度将直接从该函数的计算结果流向 `self`。

    返回：
        Tensor，其数据、shape和数据类型与输入 `self` 相同。
