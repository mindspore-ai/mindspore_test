mindspore.numpy.roll
=================================

.. py:function:: mindspore.numpy.roll(a, shift, axis=None)

    将Tensor沿给定的轴进行滚动。
    滚动超出最后位置的元素会重新回到第一个位置。

    参数：
        - **a** (Tensor) - 输入Tensor。
        - **shift** (Union[int, tuple(int)]) - 元素滚动的位移数量。如果为tuple，则 ``axis`` 必须是相同大小的tuple，并且每个给定的轴按对应的位移数量滚动。如果 ``shift`` 是整数，而 ``axis`` 是整数tuple，则所有给定的轴都使用相同的位移值。
        - **axis** (Union[int, tuple(int)], 可选) - 沿哪个轴或哪些轴滚动元素。默认情况下，数组在滚动前会被展平，然后恢复原始shape。默认值: ``None`` 。

    返回：
        Tensor，shape与 ``a`` 相同的Tensor。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果 ``axis`` 超过 ``a.ndim`` ，或者 ``shift`` 和 ``axis`` 不能广播。