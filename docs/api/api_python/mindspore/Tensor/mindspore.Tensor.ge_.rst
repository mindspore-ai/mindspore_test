mindspore.Tensor.ge\_
=====================

.. py:method:: mindspore.Tensor.ge_(other)

    逐元素计算 :math:`self >= other` 的值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        当 `self` 和 `other` 具有不同的shape时， `other` 必须能被广播成 `self` 。

    参数：
        - **other** (Union[Tensor, Number, bool]) - 第二个输入，为数值型，或bool，或数据类型为数值型或bool的Tensor。

    返回：
        Tensor，shape与 `self` 的shape相同，数据类型和 `self` 的数据类型相同。

    异常：
        - **RuntimeError** - 如果 `other` 不能被广播成 `self` 。
