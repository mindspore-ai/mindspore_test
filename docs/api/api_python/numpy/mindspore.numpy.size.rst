mindspore.numpy.size
=================================

.. py:function:: mindspore.numpy.size(a, axis=None)

    返回沿给定轴的元素数量。

    参数：
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 输入数据。
        - **axis** (int，可选) - 计算元素数量的轴。默认值： ``None`` 。如果为None，则返回总的元素数量。

    返回：
        沿指定轴的元素数量。

    异常：
        - **TypeError** - 如果输入不是类似数组的类型或 ``axis`` 不是整数。
        - **ValueError** - 如果任意轴超出范围或存在重复的轴。