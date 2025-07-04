mindspore.mint.randn
====================

.. py:function:: mindspore.mint.randn(*size, generator=None, dtype=None)

    返回一个Tensor，shape和dtype由输入决定，其元素为服从标准正态分布的数字。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **size** (Union[int, tuple(int), list(int)]) - 输出的Tensor的shape，例如，:math:`(2, 3)` or :math:`2`。

    关键字参数：
        - **generator** (:class:`mindspore.Generator`, 可选) - 伪随机数生成器。默认值： ``None`` ，使用默认伪随机数生成器。
        - **dtype** (:class:`mindspore.dtype`，可选) - 需求的输出Tensor的dtype。如果是None， `mindspore.float32` 会被使用。默认值： ``None`` 。

    返回：
        Tensor，shape和dtype由输入决定其元素为服从标准正态分布的数字。

    异常：
        - **ValueError** - 如果 `size` 包含负数。
