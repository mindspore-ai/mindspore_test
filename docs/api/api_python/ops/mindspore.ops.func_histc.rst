mindspore.ops.histc
====================

.. py:function:: mindspore.ops.histc(input, bins=100, min=0., max=0.)

    计算tensor的直方图。

    元素被分类到 `min` 和 `max` 之间的等宽箱中。
    如果 `min` 和 `max` 均为0，则使用数据的最小值和最大值。

    低于最小值和高于最大值的元素将被忽略。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **bins** (int, 可选) - 直方图箱的数量。默认 ``100`` 。
        - **min** (int, float, 可选) - 直方图数据范围的最小值。默认 ``0.`` 。
        - **max** (int, float, 可选) - 直方图数据范围的最大值。默认 ``0.`` 。

    返回：
        Tensor
