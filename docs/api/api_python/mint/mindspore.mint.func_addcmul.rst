mindspore.mint.addcmul
======================

.. py:function:: mindspore.mint.addcmul(input, tensor1, tensor2, *, value=1)

    执行 `tensor1` `tensor2` 的逐元素相乘，将结果乘以标量值 `value` ，并将其添加到 `input` 中。

    .. math::
        output[i] = input[i] + value * (tensor1[i] * tensor2[i])

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 要添加的Tensor。
        - **tensor1** (Tensor) - 要乘以的Tensor。
        - **tensor2** (Tensor) - 要乘以的Tensor。

    关键字参数：
        - **value** (Number, 可选) - tensor1和tensor2的乘数。默认值：``1``。

    返回：
        Tensor，具有与 `tensor1` * `tensor2` 相同的shape和dtype。

    异常：
        - **TypeError** - 如果 `tensor1` 、 `tensor2`、 `input` 不是Tensor。
        - **ValueError** - 如果无法将 `tensor1` 广播到 `tensor2`。
        - **ValueError** - 如果无法将 `value` 广播到 `tensor1` * `tensor2`。