mindspore.mint.mm
=================

.. py:function:: mindspore.mint.mm(input, mat2)

    计算两个矩阵的乘积。

    如果 `input` 是一个 :math:`(n \times m)` 的Tensor， `mat2` 是一个 :math:`(m \times p)` 的Tensor， `out` 则会是一个 :math:`(n \times p)` 的Tensor。

    .. note::
        此函数不能支持广播。若需要可广播的方法，请参考 :func:`mindspore.mint.matmul`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 矩阵相乘的第一个矩阵， `input` 的最后一维度必须和 `mat2` 的第一维度相等。
        - **mat2** (Tensor) - 矩阵相乘的第二个矩阵， `input` 的最后一维度必须和 `mat2` 的第一维度相等。

    返回：
        Tensor，输入的矩阵乘积。

    异常：
        - **ValueError** - `input` 的最后一维度和 `mat2` 的倒数第二维度不相等。
        - **TypeError** - `input` 或者 `mat2` 不是一个Tensor。
        - **TypeError** - `input` 或者 `mat2` 的数据类型不是float16、float32、bfloat16之一。
