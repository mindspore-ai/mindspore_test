mindspore.ops.where
====================

.. py:function:: mindspore.ops.where(condition, input, other)

    返回一个tensor，其中元素根据 `condition` 从 `input` 或 `other` 中选择。

    支持广播。

    .. math::

        output_i = \begin{cases} input_i,\quad &if\ condition_i \\ other_i,\quad &otherwise \end{cases}

    参数：
        - **condition** (Tensor[bool]) - 如果为 ``True`` ，选取 `input` 中的元素，否则选取 `other` 中的元素。
        - **input** (Union[Tensor, Scalar]) - 在 `condition` 为 ``True`` 的索引处选择的值。
        - **other** (Union[Tensor, Scalar]) - 当 `condition` 为 ``False`` 的索引处选择的值。

    返回：
        Tensor
