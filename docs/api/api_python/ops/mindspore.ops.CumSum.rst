﻿mindspore.ops.CumSum
=====================

.. py:class:: mindspore.ops.CumSum(exclusive=False, reverse=False)

    计算输入Tensor在指定轴上的累加和。

    .. math::
        y_i = x_1 + x_2 + x_3 + ... + x_i

    参数：
        - **exclusive** (bool，可选) - 表示输出结果的第一个元素是否与输入的第一个元素一致。如果为 ``False`` ，表示输出的第一个元素与输入的第一个元素一致。默认值： ``False`` 。
        - **reverse** (bool，可选) - 如果为 ``True`` ，则逆向计算累加和。默认值： ``False`` 。

    输入：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。
        - **axis** (int) - 指定要进行累加和计算的轴。仅支持常量值，该值应在[-rank(input), rank(input))范围内。

    输出：
        Tensor。输出Tensor的shape与输入Tensor的shape一致。

    异常：
        - **TypeError** - `exclusive` 或 `reverse` 不是bool。
        - **TypeError** - `axis` 不是int。