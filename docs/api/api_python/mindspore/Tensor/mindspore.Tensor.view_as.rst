mindspore.Tensor.view_as
========================

.. py:method:: mindspore.Tensor.view_as(other) -> Tensor

    根据 `other` 的shape改变原 `self` Tensor的shape。

    参数：
        - **other** (Tensor) - 返回Tensor的shape与 `other` 的shape一致。

    返回：
        Tensor，和 `other` 具有相同的shape。

    异常：
        - **TypeError** - `other` 不是Tensor。
