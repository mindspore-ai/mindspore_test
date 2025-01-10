mindspore.Tensor.log10
======================

.. py:method:: mindspore.Tensor.log10()

    逐元素返回Tensor以10为底的对数。

    .. math::
        y_i = \log_{10}(x_i)

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 如果输入值在(0, 0.01]或[0.95, 1.05]范围内，则输出精度可能会存在误差。

    .. note::
        - 输入的值必须大于0。

    返回：
        Tensor，具有与 `self` 相同的shape，dtype根据 `self.dtype` 而变化。
        
        - 如果 `self.dtype` 为float16、float32、float64、bfloat16、complex64或complex128，输出数据类型与 `self.dtype` 相同。
        - 如果 `self.dtype` 为double，输出类型为float64。
        - 在Ascend平台上，如果 `self.dtype` 为integer或boolean，输出数据类型为float32。

    异常：
        - **TypeError** - `self.dtype` 不是bool、uint8、int8、int16、int32、int64、bfloat16、float16、float32、float64、
          double、complex64、complex128其中一种。
        - **TypeError** - 在CPU或GPU平台上，如果 `self.dtype` 是integer或者boolean。
