mindspore.ops.acosh
====================

.. py:function:: mindspore.ops.acosh(input)

    逐元素计算输入tensor的反双曲余弦。

    .. math::
        out_i = \cosh^{-1}(input_i)

    .. note::
        给定一个输入tensor `input` ，该函数计算每个元素的反双曲余弦。输入范围为[1, inf]。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
