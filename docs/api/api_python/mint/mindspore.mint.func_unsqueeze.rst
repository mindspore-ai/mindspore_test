mindspore.mint.unsqueeze
========================

.. py:function:: mindspore.mint.unsqueeze(input, dim) -> Tensor

    返回一个新Tensor，并在其指定位置插入一个大小为1的维度。

    返回的张量与 `input` 共享相同的底层数据。

    `dim` 的取值范围在 :math:`[-input.dim() - 1, input.dim() + 1)` 内。负数的 `dim` 将在 :math:`dim = dim + input.dim() + 1` 的位置插入新维度。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dim** (int) - 新插入的维度的位置。

    返回：
        Tensor，若 `dim` 是0，那么它的shape为 :math:`(1, n_1, n_2, ..., n_R)`。它与 `input` 的数据类型相同。
