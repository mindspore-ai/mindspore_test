mindspore.numpy.gradient
========================

.. py:function:: mindspore.numpy.gradient(f, *varargs, axis=None, edge_order=1)

    返回一个N维数组的梯度。
    该梯度是通过在内部点使用二阶精确的中心差分法，在边界使用一阶或二阶精确的单侧(前向或后向)差分法计算得出的。
    因此返回的梯度与输入数组具有相同shape。

    .. note::
        目前我们仅支持 `edge_order` =1 且 `varargs` 间距均匀。

    参数：
        - **f** (Union[tuple, list, Tensor]) - 一个包含标量函数样本的N维数组。
        - **varargs** (Union[tuple[number], tuple[tensor scalar]], 可选) - f值之间的间距。在所有维度下默认为单位间距。间距可以使用以下方式指定： 1.单个标量指定所有维度的样本距离。 2.N个标量指定每个维度的恒定样本距离。
        - **axis** (Union[None, int, tuple(int), list(int)], 可选) -  梯度只沿给定的单个或多个轴计算。 默认情况下 ``(axis = None)`` 计算输入Tensor的所有轴的梯度。 `axis` 可能为负，在这种情况下，倒数计数。
        - **edge_order** (int, 可选) - 梯度使用N阶边界差分计算。 默认值： ``1`` 。

    返回：
        梯度，元素为Tensor的list(如果只有一个维度需要计算，则为单个Tensor)。 每个导数的shape与 `f` 相同。

    异常：
        - **TypeError** - 如果输入类型不符合上述规定。
        - **ValueError** - 如果 `axis` 超出范围，或 `f` 的shape中有维度小于1
        - **NotImplementedError** - 如果 `edge_order` != 1，或 `varargs` 中包含非标量条目。