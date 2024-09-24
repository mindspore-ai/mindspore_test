mindspore.numpy.cross
=====================

.. py:function:: mindspore.numpy.cross(a, b, axisa=- 1, axisb=- 1, axisc=- 1, axis=None)

    返回两个向量(数组)的叉积。

    在 :math:`R^3` 中，叉积是垂直于 `a` 和 `b` 的一个向量。 如果 `a` 和 `b` 是向量数组，向量默认由 `a` 和 `b` 的最后一个轴定义，这些轴可以为二或三维。 当 `a` 和 `b` 其中任意一个为二维时，输入向量的第三个分量将被假设为零，并相应计算叉积。 在两个输入向量都为二维的情况下，返回叉积的z-分量。

    参数：
        - **a** (Union[list, tuple, Tensor]) - 第一个向量的分量。
        - **b** (Union[list, tuple, Tensor]) - 第二个向量的分量。
        - **axisa** (int, 可选) - 定义 `a` 中用于计算的向量的轴。 默认为最后一个轴。
        - **axisb** (int, 可选) - 定义 `b` 中用于计算的向量的轴。 默认为最后一个轴。
        - **axisc** (int, 可选) - 定义 `c` 中包含叉积结果的轴。 如果两个输入向量都为二维，此项被忽略。 默认为最后一个轴。
        - **axis** (int, 可选) - 如果定义该参数， `axis` 将定义 `a` 和 `b` 中用于计算的向量的轴，以及 `c` 中包含叉积结果的轴。 即覆盖 `axisa` 、 `axisb` 和 `axisc` 。 默认值： ``None`` 。

    返回：
        Tensor，向量叉积。

    异常：
        - **ValueError** - 当向量的维数不是2或3时。
