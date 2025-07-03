mindspore.ops.mm
=================

.. py:function:: mindspore.ops.mm(input, mat2)

    计算两个输入的矩阵乘积。

    若 `input` 的shape为 :math:`(n \times m)` ， `mat2` 的shape为 :math:`(m \times p)` ， `out` 的shape为 :math:`(n \times p)` 。

    .. note::
        - 此函数不能支持广播。若需要可广播的方法，请参考 :func:`mindspore.ops.matmul`。
        - Ascend平台，不支持float64类型。

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **mat2** (Tensor) - 第二个输入tensor。

    返回：
        Tensor
