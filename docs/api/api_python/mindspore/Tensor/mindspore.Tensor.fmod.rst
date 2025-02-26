mindspore.Tensor.fmod
=====================

.. py:method:: mindspore.Tensor.fmod(other)

    计算除法运算self/other的浮点余数。

    .. math::
        out = self - n * other

    其中 :math:`n` 是 :math:`self/other` 结果中的整数部分。
    返回值的符号和 `self` 相同，在数值上小于 `other` 。

    参数：
        - **other** (Union[Tensor, Number]) - 除数。

    返回：
        Tensor，输出的shape与广播后的shape相同，数据类型取两个输入中精度较高或位数较多的。

    异常：
        - **TypeError** - `self` 和 `other` 都不是Tensor。
