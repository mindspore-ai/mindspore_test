mindspore.Tensor.sigmoid
=============================

.. py:method:: mindspore.Tensor.sigmoid()

    逐元素计算Sigmoid激活函数。Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)}

    其中 :math:`x_i` 是 `x` 的一个元素。

    Sigmoid函数图:

    .. image:: ../../images/Sigmoid.png
        :align: center

    返回：
        Tensor。数据类型和shape与输入相同。

    异常：
        - **TypeError** - 如果输入数据类型不是float16、float32、float64、complex64或者complex128。
        - **TypeError** - 如果输入不是Tensor。
