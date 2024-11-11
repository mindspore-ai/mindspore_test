mindspore.mint.log2
===================

.. py:function:: mindspore.mint.log2(input)

    逐元素返回Tensor以2为底的对数。

    .. math::
        y_i = \log_2(x_i)

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 如果输入值在(0, 0.01]或[0.95, 1.05]范围内，则输出精度可能会存在误差。

    参数：
        - **input** (Tensor) - 任意维度的输入Tensor。其值必须大于0。

    返回：
        Tensor，具有与 `input` 相同的shape，dtype根据 `input.dtype` 变化。
        
        - 如果 `input.dtype` 为 [float16、float32、float64、bfloat16]，输出数据类型dtype与输入 `input.dtype` 相同。
        - 如果 `input.dtype` 为整数或布尔类型，输出数据类型dtype为float32。

    异常：
        - **TypeError** - `input` 不是Tensor。