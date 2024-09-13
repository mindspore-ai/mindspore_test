mindspore.numpy.flip
=================================

.. py:function:: mindspore.numpy.flip(m, axis=None)

    沿给定轴反转数组中的元素顺序。
    数组的shape保持不变，但元素的顺序会被重新排列。

    参数：
        - **m** (Tensor) - 输入数组。
        - **axis** (Union[int, tuple(int), None], 可选) - 反转的轴或轴的tuple。默认值： ``axis=None`` ，表示对输入数组的所有轴进行反转。如果 ``axis`` 为负数，则从最后一个轴开始计数。如果 ``axis`` 是整数tuple，则对tuple中指定的所有轴进行反转。

    返回：
        Tensor，沿指定 ``axis`` 的元素被反转。

    异常：
        - **TypeError** - 如果输入不是Tensor。