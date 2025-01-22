mindspore.mint.nn.functional.softsign
=====================================

.. py:function:: mindspore.mint.nn.functional.softsign(input)

    SoftSign激活函数。

    SoftSign函数定义为：

    .. math::
        \text{SoftSign}(x) = \frac{x}{1 + |x|}

    Softsign激活函数图：

    .. image:: ../images/Softsign.png
        :align: center

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - Softsign的输入。

    返回：
        Tensor，shape与 `input` 相同。当输入数据类型为bool、int8、uint8、int16、int32、int64时，返回值数据类型为float32。否则，返回值数据类型与输入数据类型相同。
