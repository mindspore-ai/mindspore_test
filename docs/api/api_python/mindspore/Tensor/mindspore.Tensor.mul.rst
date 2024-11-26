mindspore.Tensor.mul
==========================

.. py:method:: mindspore.Tensor.mul(value)

    两个Tensor逐元素相乘。

    .. math::

        out_{i} = tensor_{i} * other_{i}

    .. note::
        - 当两个输入具有不同的shape时，它们的shape必须要能广播为一个共同的shape。
        - `self` 和 `other` 不能同时为bool类型。[True, Tensor(True, bool\_), Tensor(np.array([True]), bool\_)]等都为bool类型。
        - `self` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。

    参数：
        - **other** (Union[Tensor, number.Number, bool]) - `other` 可以是number.Number或bool，也可以是数据类型为number.Number或bool的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为 `self` 和 `other` 中精度较高的类型。

    异常：
        - **TypeError** - `other` 不是Tensor、number.Number或bool。
        - **ValueError** - `self` 和 `other` 的shape不相同。
