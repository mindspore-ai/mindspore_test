mindspore.Tensor.reshape
========================

.. py:method:: mindspore.Tensor.reshape(*shape)

    基于给定的 `shape` ，对当前Tensor进行重新排列。

    `shape` 最多只能有一个-1，在这种情况下，它可以从剩余的维度和当前Tensor的元素个数中推断出来。

    参数：
        - **shape** (Union[tuple[int], list[int], Tensor[int]]) - 如果 `shape` 是list或者tuple，其元素需为整数，并且只支持常量值，如 :math:`(y_1, y_2, ..., y_S)` ；如果 `shape` 是Tensor，数据类型必须为int32或者int64，并且只支持一维Tensor。

    返回：
        Tensor，若给定的 `shape` 中不包含-1, 则输出 `shape` 为 :math:`(y_1, y_2, ..., y_S)` ；若给定的 `shape` 中第 `k` 个位置为-1，则输出 `shape` 为 :math:`(y_1, ..., y_{k-1}, \frac{\prod_{i=1}^{R}x_{i}}{y_1\times ...\times y_{k-1}\times y_{k+1}\times...\times y_S} , y_{k+1},..., y_S)`，其中输入Tensor的 `shape` 为 :math:`(x_1, x_2, ..., x_R)` 。

    异常：
        - **ValueError** - 如果 `shape` 包含超过一个-1。
        - **ValueError** - 如果 `shape` 包含的元素小于-1。
        - **ValueError** - 针对于 `shape` 中不包含-1的场景，如果 `shape` 的元素总数不等于当前Tensor的元素总数，:math:`\prod_{i=1}^{R}x_{i} \ne \prod_{i=1}^{S}y_{i}` （即不匹配当前Tensor的数组大小）；针对于 `shape` 中包含-1的场景，如果除去 `shape` 中的-1外，其他元素总数无法被当前Tensor的元素总数整除。
