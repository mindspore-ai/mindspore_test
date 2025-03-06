mindspore.ops.where
====================

.. py:function:: mindspore.ops.where(condition, input, other)

    返回一个Tensor，Tensor的元素从 `input` 或 `other` 中根据 `condition` 选择。

    .. math::

        output_i = \begin{cases} input_i,\quad &if\ condition_i \\ other_i,\quad &otherwise \end{cases}

    参数：
        - **condition** (Tensor[bool]) - 如果是 ``True`` ，选取 `input` 中的元素，否则选取 `other` 中的元素。
        - **input** (Union[Tensor, Scalar]) - 在 `condition` 为 ``True`` 的索引处选择的值。
        - **other** (Union[Tensor, Scalar]) - 当 `condition` 为 ``False`` 的索引处选择的值。

    返回：
        Tensor，其中的元素从 `input` 和 `other` 中选取。

    异常：
        - **TypeError** - `condition` 不是Tensor。
        - **TypeError** - `input` 和 `other` 都是常量。
        - **ValueError** - `condition` 、 `input` 和 `other` 不能互相广播。
