mindspore.Tensor.unsqueeze
============================

.. py:method:: mindspore.Tensor.unsqueeze(dim)

    对输入 `self` 在给定维上添加额外维度。

    扩展后的Tensor中位置对应 `dim` 的维度为插入的新维度。

    参数：
        - **dim** (int) - 新插入的维度的位置。 `dim` 的值必须在范围 `[-self.ndim-1, self.ndim]` 内。仅接受常量输入。

    返回：
        Tensor，维度在指定轴扩展之后的Tensor，与 `self` 的数据类型相同。如果 `dim` 是0，那么它的shape为 :math:`(1, n_1, n_2, ..., n_R)`。

    异常：
        - **TypeError** - 如果 `dim` 不是int。
        - **ValueError** - 如果 `dim` 超出了 :math:`[-self.ndim-1, self.ndim]` 的范围。
