mindspore.Tensor.sqrt\_
=======================

.. py:method:: mindspore.Tensor.sqrt_()

    将当前Tensor `self` 中的每个元素替换为其平方根。

    .. note::
        当 `self` 的某个元素为负数，则该位置上的结果取决于平台：

        - 在Atlas训练系列产品上将计算其相反数的平方根。
        - 在其他平台（如Atlas A2训练系列产品）上将得到 ``NaN`` 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. math::
        self_{i} =  \sqrt{self_{i}}

    返回：
        Tensor，返回被修改后的 `self` 自身。
