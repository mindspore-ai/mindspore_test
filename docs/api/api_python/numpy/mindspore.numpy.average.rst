mindspore.numpy.average
=======================

.. py:function:: mindspore.numpy.average(x, axis=None, weights=None, returned=False)

    沿指定轴计算加权平均值。

    参数：
        - **x** (Tensor) - 需要平均的Tensor。
        - **axis** (Union[None, int, tuple(int)]，可选) - 沿 `axis` 对 `x` 进行平均。默认值： ``None`` 。 如果 `axis` 为 `None` ，它将对张量 `x` 的所有元素进行平均。如果 `axis` 是负数，它将从最后一个 `axis` 数回到第一个 `axis` 。
        - **weights** (Union[None, Tensor]，可选) - `weights` 与 `x` 中的值相关联。 默认值： ``None`` 。 如果 `weights` 为 `None` ，所有 `x` 中的数据的权重假设为1。 如果 `weights` 是一个1-D的Tensor， 其长度必须与给定的 `axis` 的长度相等。 否则， `weights` 应该与 `x` 具有相同的shape。
        -  **returned** (bool，可选) - 默认值：`False`。 如果为 `True`， 函数将返回tuple (average, sum_of_weights)。 如果为 `False` ，只返回平均值。

    返回：
        平均后的Tensor。 如果 `returned` 为 `True` ，返回tuple。