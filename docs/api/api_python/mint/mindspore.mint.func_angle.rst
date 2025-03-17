mindspore.mint.angle
=====================

.. py:function:: mindspore.mint.angle(input)

    逐元素计算 `input` 张量中每个元素的幅角（以弧度表示）。

    .. math::

        out_i = angle(input_i)

    .. warning::
        - 这是一个实验性API，后续可能修改或删除；
        - 若输入 `input` 的数据类型为复数形式，该算子不支持反向求导。

    .. note::
        该函数对于负实数返回 ``Pi`` 弧度，对非负实数则返回零弧度。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，当输入类型为float16时，返回类型也为float16；其余的输入类型则返回float32。
        返回shape与输入相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 的数据类型为float64、complex128等不支持的类型。
