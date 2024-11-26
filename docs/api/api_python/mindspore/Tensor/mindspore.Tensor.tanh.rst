mindspore.Tensor.tanh
=====================

.. py:method:: mindspore.Tensor.tanh()

    逐元素计算自身元素的双曲正切。Tanh函数定义为：

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    其中 :math:`x_i` 是输入Tensor的元素。

    Tanh函数图：

    .. image:: ../../images/Tanh.png
        :align: center

    返回：
        Tensor，数据类型和shape与 `self` 相同。

    异常：
        - **TypeError** - `self` 不是Tensor。
