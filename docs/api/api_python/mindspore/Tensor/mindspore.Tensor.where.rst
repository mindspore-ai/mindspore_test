mindspore.Tensor.where
=======================

.. py:method:: mindspore.Tensor.where(condition, y)

    返回一个Tensor，Tensor的元素从 `self` 或 `y` 中根据 `condition` 选择。

    .. math::

        output_i = \begin{cases} self_i,\quad &if\ condition_i \\ y_i,\quad &otherwise \end{cases}

    参数：
        - **condition** (Tensor[bool]) - 如果是 ``True`` ，选取 `self` 中的元素，否则选取 `y` 中的元素。
        - **y** (Union[Tensor, Scalar]) - 当 `condition` 为 ``False`` 的索引处选择的值。

    返回：
        Tensor，其中的元素从 `self` 和 `y` 中选取。

    异常：
        - **TypeError** - 如果 `condition` 不是Tensor。
        - **ValueError** - `condition` 、 `self` 和 `y` 不能互相广播。
