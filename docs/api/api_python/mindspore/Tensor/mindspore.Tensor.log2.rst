mindspore.Tensor.log2
======================

.. py:method:: mindspore.Tensor.log2()

    逐元素返回Tensor以2为底的对数。

    .. math::
        y_i = \log_2(x_i)

    .. warning::
        - 如果log2的输入值范围在(0, 0.01]或[0.95, 1.05]区间，输出精度可能会受影响。

    .. note::
        - 输入的值必须大于0。

    返回：
        Tensor。具有与 `self` 相同的shape，并且类型根据 `self.dtype` 而变化。

        - 如果 `self.dtype` 的类型是float16、float32、float64、complex64、complex128，那么输出的类型则和 `self.dtype` 相同。
        - 如果 `self.dtype` 的类型是double，那么输出的类型为float64。
        - 在Ascend平台中，如果 `self.dtype` 的类型是integer或者boolean，那么输出的类型为float32。

    异常：
        - **TypeError** - 如果 `self.dtype` 的类型不是bool、int8、int32、int64、uint8、float16、float32、float64、double、complex64、complex128其中的一种。
        - **TypeError** - 在CPU或者GPU平台中，如果 `self.dtype` 的类型是integer或者boolean。
