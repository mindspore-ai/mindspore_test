mindspore.Tensor.scatter_add\_
==============================

.. py:method:: mindspore.Tensor.scatter_add_(dim, index, src)

    将 `src` 中所有的元素添加到 `self` 中 `index` 指定的索引处（属于原地更新操作）。
    其中 `dim` 控制scatter add操作的轴。
    `self` 、 `index` 、 `src` 三者的rank都必须大于或等于1。

    下面看一个三维的例子：

    .. code-block::

        self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0

        self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1

        self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

    .. warning::
        开启确定性计算后， `index` 不能传入非连续的Tensor，否则会无法获得确定性的计算结果。

    参数：
        - **dim** (int) - `self` 执行scatter_add操作的轴。取值范围是[-r, r)，其中r是 `self` 的rank。
        - **index** (Tensor) - `self` 要进行scatter_add操作的目标索引。数据类型为int32或int64，rank必须和 `self` 一致。除了 `dim` 指定的维度， `index` 的每一维的size都需要小于或等于 `self` 对应维度的size。
        - **src** (Tensor) - 指定与 `self` 进行scatter_add操作的Tensor，其数据类型与 `self` 类型相同，shape中每一维的size必须大于或等于 `index` 。

    返回：
        Tensor，shape和数据类型与输入 `self` 相同。

    异常：
        - **TypeError** - `index` 的数据类型不是int32或int64。
        - **ValueError** - `self` 、 `index` 和 `src` 中，任意一者的rank小于1。
        - **ValueError** - `self`、 `index` 和 `src` 的rank不一致。
        - **ValueError** - 除了 `dim` 指定的维度， `index` 的任意维的size大于 `self` 对应维度的size。
        - **ValueError** - `src` 任意维度size小于 `index` 对应维度的size。
