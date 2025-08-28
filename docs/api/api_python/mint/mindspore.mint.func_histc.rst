mindspore.mint.histc
====================

.. py:function:: mindspore.mint.histc(input, bins=100, min=0, max=0)

    计算Tensor的直方图。

    元素被分类到 `min` 和 `max` 之间的等宽箱中。
    如果 `min` 和 `max` 均为0，则使用数据的最小值和最大值。

    低于最小值和高于最大值的元素将被忽略。

    .. warning::
        如果 `input` 是int64，有效取值在int32范围内，超出范围可能产生精度误差。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **bins** (int, 可选) - 直方图箱的数量，可选。默认值： ``100`` 。若指定，则必须为正数。
        - **min** (int, float, 可选) - 范围下限（含），可选。默认值： ``0`` 。
        - **max** (int, float, 可选) - 范围上限（含），可选。默认值： ``0`` 。

    返回：
        1-D Tensor，shape为 :math:`(bins, )`，数据类型与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 的数据类型不支持。
        - **TypeError** - 如果属性 `min` 或 `max` 不是float类型或int类型。
        - **TypeError** - 如果属性 `bins` 不是整数。
        - **ValueError** - 如果属性 `min` > `max` 。
        - **ValueError** - 如果属性 `bins` <= 0。
