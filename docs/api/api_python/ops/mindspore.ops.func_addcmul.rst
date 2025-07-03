mindspore.ops.addcmul
======================

.. py:function:: mindspore.ops.addcmul(input, tensor1, tensor2, value=1)

    `tensor1` 与 `tensor2` 的逐元素相乘，将结果乘以标量值 `value` ，并加到 `input` 。

    .. math::
        output[i] = input[i] + value[i] * (tensor1[i] * tensor2[i])

    参数：
        - **input** (Tensor) - 输入tensor。
        - **tensor1** (Tensor) - 将被相乘的tensor1。
        - **tensor2** (Tensor) - 将被相乘的tensor2。
        - **value** (Union[Tensor, number]) - （tensor1 * tensor2）的乘数。默认 ``1`` 。

    返回：
        Tensor
