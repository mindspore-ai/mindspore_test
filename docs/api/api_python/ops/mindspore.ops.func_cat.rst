mindspore.ops.cat
==================

.. py:function:: mindspore.ops.cat(tensors, axis=0)

    在指定轴上拼接输入Tensor。

    输入为tuple或list，其中所有元素的秩相同，记为 :math:`R` 。设定拼接轴为 :math:`m` ，且满足 :math:`0 \le m < R` 。假设输入元素数量为 :math:`N` ，第 :math:`i` 个元素 :math:`t_i` 的shape为 :math:`(x_1, x_2, ..., x_{mi}, ..., x_R)` ，其中 :math:`x_{mi}` 是第 :math:`t_i` 个元素的第 :math:`m` 个维度。则输出Tensor的shape为：

    .. math::
        (x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)

    参数：
        - **tensors** (Union[tuple, list]) - 输入为Tensor组成的tuple或list。假设在这个tuple或list中有两个Tensor，即 `t1` 和 `t2` 。要在0轴方向上执行 `Concat` ，除  :math:`0` 轴外，其他轴的shape都应相等，即 :math:`t1.shape[1] = t2.shape[1], t1.shape[2] = t2.shape[2], ..., t1.shape[R-1] = t2.shape[R-1]` ，其中 :math:`R` 是Tensor的秩。
        - **axis** (int) - 表示指定的轴，取值范围是 :math:`[-R, R)` 。默认值： ``0`` 。

    返回：
        Tensor，shape为 :math:`(x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)` ，数据类型与 `tensors` 相同。

    异常：
        - **TypeError** - `axis` 不是int。
        - **ValueError** - `tensors` 中的Tensor维度不同。
        - **ValueError** - `axis` 的值不在 :math:`[-R, R)` 范围内。
        - **ValueError** - 除了 `axis` 之外， `tensors` 的shape不相同。
        - **ValueError** - `tensors` 为空tuple或list。
