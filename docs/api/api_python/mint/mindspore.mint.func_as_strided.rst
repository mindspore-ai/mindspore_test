mindspore.mint.as_strided
=========================

.. py:function:: mindspore.mint.as_strided(input, size, stride, storage_offset=None) -> Tensor

    返回大小为 `size`、步幅为 `stride` 和偏移量为 `storage_offset` 的 `input` 视图。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入张量。张量的形状为 :math:`(x_1, x_2, ..., x_R)` 。
        - **size** (tuple[int]) - 输出张量的形状。
        - **stride** (tuple[int]) - 输出张量的步幅。
        - **storage_offset** (int, 可选) - 输出张量在底层存储中的偏移量。如果为 ``None``，则输出张量的 `storage_offset`
          将与输入张量一致。默认值： ``None``。

    返回：
        Tensor，shape由 `size` 决定，dtype跟 `input.dtype` 相同。

    异常：
        - **TypeError** - `self` 不是Tensor。
