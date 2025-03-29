mindspore.ops.uniform
=======================

.. py:function:: mindspore.ops.uniform(shape, minval, maxval, seed=None, dtype=mstype.float32)

    生成服从均匀分布的随机数。

    .. note::
        广播后，任意位置上tensor的最小值必须小于最大值。

    参数：
        - **shape** (Union[tuple, Tensor]) - 指定生成随机数的shape。
        - **minval** (Tensor) - 指定生成随机值的最小值。
        - **maxval** (Tensor) - 指定生成随机值的最大值。
        - **seed** (int) - 随机数种子。默认 ``None`` 。
        - **dtype** (mindspore.dtype) - 指定数据类型。

    返回：
        Tensor
