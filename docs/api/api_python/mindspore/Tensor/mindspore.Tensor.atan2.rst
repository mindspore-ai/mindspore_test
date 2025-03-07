mindspore.Tensor.atan2
======================

.. py:method:: mindspore.Tensor.atan2(other)

    逐元素计算self/other的反正切值。

    返回 :math:`\theta\ \in\ [-\pi, \pi]` ，使得 :math:`self = r*\sin(\theta), other = r*\cos(\theta)` ，其中 :math:`r = \sqrt{self^2 + other^2}` 。

    .. note::
        - `self` 和参数 `other` 遵循隐式类型转换规则，使数据类型保持一致。如果两参数数据类型不一致，则低精度类型会被转换成较高精度类型。

    参数：
        - **other** (Tensor, Number.number) - 输入Tensor或常量，shape与 `self` 相同，或能与 `self` 的shape广播。

    返回：
        Tensor，与广播后的输入shape相同，和 `self` 数据类型相同。

    异常：
        - **TypeError** - `other` 不是Tensor或常量。
        - **RuntimeError** - `self` 与 `other` 的数据类型不支持转换。
