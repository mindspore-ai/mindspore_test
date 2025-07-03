mindspore.ops.digamma
=====================

.. py:function:: mindspore.ops.digamma(input)

    逐元素计算输入tensor的伽玛函数的对数导数。

    .. math::
        P(x) = \frac{d}{dx}(\ln (\Gamma(x)))

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。
    返回：
        Tensor