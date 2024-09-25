mindspore.numpy.rot90
=================================

.. py:function:: mindspore.numpy.rot90(a, k=1, axes=(0, 1))

    将Tensor在指定的轴平面内旋转90度。旋转方向是从第一个轴指向第二个轴。

    参数：
        - **a** (Tensor) - 输入的二维或更多维的Tensor。
        - **k** (int) - Tensor旋转90度的次数。默认值: ``1`` 。
        - **axes** (Union[tuple(int), list(int)]，可选) - Tensor在由轴定义的平面内进行旋转。默认值: ``(0, 1)`` 。轴必须不同，并且shape为 ``(2,)`` 。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入 ``a`` 不是Tensor，或参数 ``k`` 不是整数，或参数 ``axes`` 不是整数的tuple或list。
        - **ValueError** - 如果任何轴超出范围，或 ``axes`` 的长度不是2。