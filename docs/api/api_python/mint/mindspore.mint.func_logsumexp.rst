mindspore.mint.logsumexp
========================

.. py:function:: mindspore.mint.logsumexp(input, dim, keepdim=False)

    计算输入 `input` 中指定 `dim` 维度上所有元素指数和的对数（计算过程经过数值稳定处理）， 并根据 `keepdim` 参数决定是否保留该维度。

    .. math::

        logsumexp(input) = \log(\sum(e^{input-input_{max}})) + input_{max}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dim** (Union[int, tuple(int), list(int)], 可选) - 需要规约的维度（数值在 `[0, len(input.shape) - 1]` 之间），输入为 `()` 时规约所有维度。
        - **keepdim** (bool, 可选) - 输出张量是否保留维度 `dim`，默认值： ``False``。

    返回：
        Tensor，数据类型dtype根据 `input.dtype` 变化，shape根据输入 `dim` 和 `keepdim` 的数值而变化。
        - 如果 `input.dtype` 为 [float16, float32, bfloat16]，输出数据类型dtype与输入 `input.dtype` 相同。
        - 如果 `input.dtype` 为整数或布尔类型，输出数据类型dtype为float32。
        - 如果 `dim` 为 `()`，并且 `keepdim` 为 `False`，则输出为一个零维Tensor，表示输入 `input` 中所有元素指数和的对数。
        - 如果 `dim` 为 `1`，并且 `keepdim` 为 `False`，则输出shape为 :math:`(input.shape[0], input.shape[2], ..., input.shape[n])`。
        - 如果 `dim` 为 `(1, 2)`，并且 `keepdim` 为 `False`， 则输出shape为 :math:`(input.shape[0], input.shape[3], ..., input.shape[n])`。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不是bool、int8、int16、int32、int64、uint8、float16、float32、bfloat16之一。
        - **TypeError** - `dim` 不是int或tuple(int)或list(int)。
        - **TypeError** - `keepdim` 不是bool类型。
        - **ValueError** - `dim` 中任意元素的数值不在 `[0, len(input.shape) - 1]` 之间。
        - **RuntimeError** - `dim` 中任意元素重复。
