mindspore.mint.min
===================

.. py:function:: mindspore.mint.min(input)

    返回输入Tensor的最小值。

    参数：
        - **input** (Tensor) - 输入的Tensor。

    返回：
        输入Tensor的最小值，类型与 `input` 相同。

    .. py:function:: mindspore.mint.min(input, dim, keepdim=False)
        :noindex:

    在给定轴上计算输入Tensor的最小值。

    参数：
        - **input** (Tensor) - 输入任意维度的Tensor。不支持complex类型。
        - **dim** (int) - 指定计算维度。
        - **keepdim** (bool, 可选) - 表示是否减少维度，如果为 ``True`` ，输出将与输入保持相同的维度；如果为 ``False`` ，输出将减少维度。默认值： ``False`` 。

    返回：
        tuple (Tensor)，如果 `dim` 不为默认值 ``None``，表示2个Tensor组成的tuple，包含输入Tensor沿给定维度的最小值和对应的索引。

        - **values** (Tensor) - 输入Tensor沿给定维度的最小值，数据类型和 `input` 相同。如果 `keepdim` 为 ``True`` ，输出Tensor的维度是 :math:`(input_1, input_2, ...,input_{axis-1}, 1, input_{axis+1}, ..., input_N)` 。否则输出维度为 :math:`(input_1, input_2, ...,input_{axis-1}, input_{axis+1}, ..., input_N)` 。
        - **index** (Tensor) - 输入Tensor沿给定维度最小值的索引，shape和 `values` 相同。

    异常：
        - **ValueError** - 如果 `dim` 为默认值 ``None`` 且 `keepdim` 不是 ``False``。