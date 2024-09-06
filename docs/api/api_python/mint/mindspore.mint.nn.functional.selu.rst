mindspore.mint.nn.functional.selu
===================================

.. py:function:: mindspore.mint.nn.functional.selu(input)

    激活函数selu（Scaled exponential Linear Unit）。

    该激活函数定义为：

    .. math::
        E_{i} =
        scale *
        \begin{cases}
        x_{i}, &\text{if } x_{i} \geq 0; \cr
        \text{alpha} * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    其中， :math:`alpha` 和 :math:`scale` 是预定义的常量（ :math:`alpha=1.67326324` ， :math:`scale=1.05070098` ）。

    更多详细信息，请参见 `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_ 。

    SELU函数图：

    .. image:: ../images/SeLU.png
        :align: center

    参数：
        - **input** (Tensor) - 任意维度的Tensor，数据类型为float16、float32、bfloat16。

    返回：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 的数据类型不是float16、float32、bfloat16。
