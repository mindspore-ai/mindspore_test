mindspore.Tensor.cummin
=======================

.. py:method:: mindspore.Tensor.cummin(dim)

    返回一个由 `(values, indices)` 组成的Tuple，其中 `values` 是当前Tensor沿维度 `dim` 的累积最小值， `indices` 是每个最小值的索引位置。

    .. math::
        \begin{array}{ll} \\
            y_{i} = \min(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    .. note::
        - Ascend不支持O2模式。
        - 仅在Ascend上支持梯度计算。

    参数：
        - **dim** (int) - 算子操作的维度，维度的大小范围是 `[-self.ndim, self.ndim - 1]` 。

    返回：
        - **values** (Tensor) - 累积最小值Tensor，数据类型及shape与当前Tensor相同。
        - **indices** (Tensor) - `values` 的每一个元素在当前Tensor的对应行中的索引，数据类型为int64，shape与当前Tensor相同。

    异常：
        - **ValueError** - 如果 `dim` 不在范围 `[-self.ndim, self.ndim - 1]` 内。
