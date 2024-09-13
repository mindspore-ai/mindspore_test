mindspore.numpy.sometrue
=================================

.. py:function:: mindspore.numpy.sometrue(a, axis=None, keepdims=False)

    测试沿给定轴是否有任意数组元素为True。
    默认返回单一boolean，除非 ``axis`` 不为 ``None`` 。

    参数：
        - **a** (Union[int, float, bool, list, tuple, Tensor]) - 输入Tensor或可以转换为数组的对象。
        - **axis** (Union[None, int, tuple(int)]) - 执行逻辑或（OR）操作的轴或轴的tuple。默认值: ``None`` 。如果为None，则对输入数组的所有维度进行逻辑或（OR）操作。如果为负数，则从最后一个轴到第一个轴进行计算。如果为整数tuple，则在多个轴上进行操作，而不是单个轴或所有轴。
        - **keepdims** (bool) - 默认值: ``False`` 。如果为True，则保留被处理的轴作为大小为1的维度。启用此选项后，结果将能够与输入数组正确广播。如果传递了默认值，则 ``keepdims`` 不会传递给ndarray子类的任何方法，但任何非默认值会传递。如果子类方法不实现 ``keepdims`` ，可能会引发异常。

    返回：
        返回单一boolean，除非 ``axis`` 不为 ``None`` 。

    异常：
        - **TypeError** - 如果输入不是类似数组的对象，或者 ``axis`` 不是整数或整数tuple，或者 ``keepdims`` 不是bool类型。
        - **ValueError** - 如果任何轴超出范围或存在重复轴。