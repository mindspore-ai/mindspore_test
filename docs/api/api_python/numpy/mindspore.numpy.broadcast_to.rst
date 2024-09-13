mindspore.numpy.broadcast_to
=================================

.. py:function:: mindspore.numpy.broadcast_to(array, shape)

    将数组广播到新shape。

    参数：
        - **array** (Tensor) - 要广播的数组。
        - **shape** (tuple) - 目标数组的shape。

    返回：
        Tensor，原始数组被广播到指定的shape。

    异常：
        - **ValueError** - 如果数组不能被广播到指定shape。