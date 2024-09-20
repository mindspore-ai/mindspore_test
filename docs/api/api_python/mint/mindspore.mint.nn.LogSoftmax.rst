mindspore.mint.nn.LogSoftmax
============================

.. py:class:: mindspore.mint.nn.LogSoftmax(dim=None)

    在指定轴上对输入Tensor应用LogSoftmax函数。假设在指定轴上， :math:`x` 对应每个元素 :math:`x_i` ，则LogSoftmax函数如下所示：

    .. math::
        \text{output}(x_i) = \log \left(\frac{\exp(x_i)} {\sum_{j = 0}^{N-1}\exp(x_j)}\right),

    其中， :math:`N` 为Tensor长度。

    参数：
        - **dim** (int, 可选) - 指定进行Log softmax运算的轴。默认值： ``None`` 。

    返回：
        Tensor，和输入Tensor的shape相同。

    异常：
        - **ValueError** - 如果 `dim` 超出范围[-len(input.shape), len(input.shape))。
