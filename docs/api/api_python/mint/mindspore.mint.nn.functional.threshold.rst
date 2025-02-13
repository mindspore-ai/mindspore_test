mindspore.mint.nn.functional.threshold
======================================

.. py:function:: mindspore.mint.nn.functional.threshold(input, threshold, value, inplace=False)

    逐元素计算Threshold激活函数。

    Threshold定义为：

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **threshold** (Union[int, float]) - 阈值。
        - **value** (Union[int, float]) - 小于阈值时的填充值。
        - **inplace** (bool, 可选) - 是否启用原地更新功能。默认值： ``False`` 。

    返回：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `threshold` 不是浮点数或整数。
        - **TypeError** - `value` 不是浮点数或整数。
