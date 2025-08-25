mindspore.mint.nn.functional.relu6
==================================

.. py:function:: mindspore.mint.nn.functional.relu6(input, inplace=False)

    逐元素计算输入Tensor的ReLU（修正线性单元），其上限为6。

    .. math::
        \text{ReLU6}(input) = \min(\max(0,input), 6)

    返回 :math:`\min(\max(0,input), 6)` 元素的值。

    ReLU6函数图：

    .. image:: ../images/ReLU6.png
        :align: center

    参数：
        - **input** (Tensor) - 输入Tensor。数据类型为int8、int16、int32、int64、uint8、float16、float32、bfloat16。
        - **inplace** (bool, 可选) - 是否采用原地更新模式，默认值为 ``False``。

    返回：
        Tensor，其shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不是int8、int16、int32、int64、uint8、float16、float32、bfloat16。
