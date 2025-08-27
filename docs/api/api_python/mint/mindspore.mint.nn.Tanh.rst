mindspore.mint.nn.Tanh
======================

.. py:class:: mindspore.mint.nn.Tanh

    逐元素计算Tanh（双曲正切值）激活函数，返回一个新的Tensor。

    Tanh函数定义为：

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    其中 :math:`x_i` 是输入Tensor的元素。

    Tanh函数图：

    .. image:: ../images/Tanh.png
        :align: center

    输入：
        - **input** (Tensor) - 任意维度的Tensor，数据类型为float16或float32的输入。

    输出：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 的数据类型既不是float16也不是float32。
