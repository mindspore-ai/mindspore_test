mindspore.numpy.array_equiv
=================================

.. py:function:: mindspore.numpy.array_equiv(a1, a2)

    当输入数组shape一致且所有元素相等时，返回 ``True`` 。
    shape一致意味着它们要么具有相同的shape，要么可以通过广播使其中一个输入数组与另一个具有相同的shape。

    .. note::
        在MindSpore中，会返回一个bool tensor，因为在图模式下，该值无法在编译时追踪和计算。

    参数：
        - **a1/a2** (Union[int, float, bool, list, tuple, Tensor]) - 输入数组。

    返回：
        Scalar bool tensor，如果输入等效，值为 ``True`` ，否则为 ``False`` 。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。