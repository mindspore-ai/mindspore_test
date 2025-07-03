mindspore.mint.nn.Threshold
===========================

.. py:class:: mindspore.mint.nn.Threshold(threshold, value, inplace=False)

    逐元素计算Threshold激活函数。

    Threshold定义为：

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    参数：
        - **threshold** (Union[int, float]) - 阈值。
        - **value** (Union[int, float]) - 小于阈值时的填充值。
        - **inplace** (bool, 可选) - 是否启用原地更新功能。默认值： ``False`` 。

    输入：
        - **input** (Tensor) - 输入Tensor。

    输出：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `threshold` 不是浮点数或整数。
        - **TypeError** - `value` 不是浮点数或整数。
