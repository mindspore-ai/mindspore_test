mindspore.mint.nn.LogSigmoid
=============================

.. py:class:: mindspore.mint.nn.LogSigmoid

    逐元素计算LogSigmoid激活函数。输入是任意格式的Tensor。

    LogSigmoid定义为：

    .. math::
        \text{LogSigmoid}(x_{i}) = \log(\frac{1}{1 + \exp(-x_i)}),

    其中，:math:`x_{i}` 是输入Tensor的一个元素。

    LogSigmoid函数图：

    .. image:: ../images/LogSigmoid.png
        :align: center

    输入：
        - **input** (Tensor) - LogSigmoid的输入，数据类型为bfloat16，float16或float32。shape为 :math:`(*)` ，其中 :math:`*` 表示任意的附加维度。

    输出：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 的数据类型既不是bfloat16，float16也不是float32。
        - **TypeError** - `input` 不是一个Tensor。
