mindspore.mint.special.exp2
=============================

.. py:function:: mindspore.mint.special.exp2(input)

    逐元素计算Tensor `input` 以2为底的指数。

    .. math::
        out_i = 2^{input_i}

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
