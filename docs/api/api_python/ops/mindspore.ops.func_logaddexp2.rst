mindspore.ops.logaddexp2
========================

.. py:function:: mindspore.ops.logaddexp2(input, other)

    计算以2为底的输入的指数和的对数。

    .. math::

        out_i = \log_2(2^{input_i} + 2^{other_i})

    参数：
        - **input** (Tensor) - 输入tensor。
        - **other** (Tensor) - 输入tensor。

    返回：
        Tensor
