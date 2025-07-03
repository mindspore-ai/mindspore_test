mindspore.mint.min
===================

.. py:function:: mindspore.mint.min(input) -> Tensor

    返回输入Tensor的最小值。

    参数：
        - **input** (Tensor) - 输入的Tensor。

    返回：
        Tensor，值为输入Tensor的最小值，类型与 `input` 相同。

    .. py:function:: mindspore.mint.min(input, dim, keepdim=False) -> Tensor
        :noindex:

    在给定轴上计算输入Tensor的最小值，并返回最小值和索引值。

    参数：
        - **input** (Tensor) - 输入任意维度的Tensor，设置输入Tensor的shape为 :math:`(input_1, input_2, ..., input_N)` ，不支持complex类型。
        - **dim** (int) - 指定计算维度。
        - **keepdim** (bool, 可选) - 表示是否减少维度，如果为 ``True`` ，输出将与输入保持相同的维度；如果为 ``False`` ，输出将减少维度。默认值： ``False`` 。

    返回：
        tuple(Tensor)，返回两个元素类型为Tensor的tuple，包含输入Tensor沿指定维度 `dim` 的最小值和相应的索引。

        - **values** (Tensor) - 输入Tensor沿给定维度的最小值，shape和 `index` 相同，数据类型和 `self` 相同。
        - **index** (Tensor) - 输入Tensor的沿给定维度的最小值索引，数据类型为 `int64` 。如果 `keepdim` 为 ``True`` ，输出Tensor的维度是 :math:`(input_1, input_2, ...,input_{dim-1}, 1, input_{dim+1}, ..., input_N)` 。否则输出维度为 :math:`(input_1, input_2, ...,input_{dim-1}, input_{dim+1}, ..., input_N)` 。

    异常：
        - **TypeError** - 如果 `input` 不是tensor。
        - **TypeError** - 如果 `keepdim` 不是bool类型。
        - **TypeError** - 如果 `dim` 不是int类型。

    .. py:function:: mindspore.mint.min(input, other) -> Tensor
        :noindex:

    详情请参考 :func:`mindspore.mint.minimum`。
