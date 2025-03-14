mindspore.ops.addcmul
======================

.. py:function:: mindspore.ops.addcmul(input, tensor1, tensor2, value=1)

    执行Tensor `tensor1` 与Tensor `tensor2` 的逐元素乘积，将结果乘以标量值 `value` ，并将其添加到 `input` 中。

    .. math::
        output[i] = input[i] + value[i] * (tensor1[i] * tensor2[i])

    参数：
        - **input** (Tensor) - 待累加的输入Tensor。
        - **tensor1** (Tensor) - 第一个待乘的Tensor。
        - **tensor2** (Tensor) - 第二个待乘的Tensor。
        - **value** (Union[Tensor, Number]，可选) - tensor1 * tensor2的乘数。默认值： ``1`` 。
        
    返回：
        Tensor，具有与tensor1*tensor2相同的shape和dtype。

    异常：
        - **TypeError** - 如果 `tensor1` 、 `tensor2`或 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 的dtype不是float32、float16或int32之一。
        - **TypeError** - 如果 `tensor1` 或 `tensor2` 的dtype不是float32、float16或int32之一。
        - **TypeError** - 如果 `value` 的dtype不是float32、float16或int32之一。
        - **ValueError** - 如果无法将 `tensor1` 广播到 `tensor2` 。
        - **ValueError** - 如果无法将 `value` 广播到 `tensor1` * `tensor2` 。
        - **ValueError** - 如果无法将 `input` 广播到 `value*(tensor1*tensor2)` 。
