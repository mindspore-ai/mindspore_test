mindspore.Tensor.gcd
====================

.. py:method:: mindspore.Tensor.gcd(other)

    按元素计算输入Tensor的最大公约数。
    两个输入的shape应该是可广播的，它们的数据类型应该是int16（使用Ascend后端时支持，GRAPH模式只在图编译等级为O0时支持）、int32或int64之一。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **other** (Tensor) - 另一个输入。

    返回：
        Tensor，返回的shape与广播后的shape，数据类型为两个输入中数字精度较高的类型。

    异常：
        - **TypeError** - 如果 `self` 或 `other` 的数据类型既不是int32也不是int64。
        - **ValueError** - 如果两个输入的shape不可广播。
