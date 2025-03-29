mindspore.ops.logsumexp
=======================

.. py:function:: mindspore.ops.logsumexp(input, dim, keepdim=False)

    计算tensor在指定维度上的指数和的对数。

    .. math::

        logsumexp(input) = \log(\sum(e^{input-input_{max}})) + input_{max}

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (Union[int, tuple(int), list(int)]) - 指定维度。如果为 ``()`` ，计算 `input` 中的所有元素。
        - **keepdim** (bool，可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
