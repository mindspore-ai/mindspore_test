mindspore.ops.SoftShrink
========================

.. py:class:: mindspore.ops.SoftShrink(lambd=0.5)

    Soft Shrink激活函数。

    更多参考详见 :func:`mindspore.ops.softshrink`。

    参数：
        - **input** (Tensor) - Soft Shrink的输入。支持数据类型：

          - Ascend：float16、float32、bfloat16。
        - **lambd** (number，可选) - Soft Shrink公式定义的阈值 :math:`\lambda` ，必须不小于零。默认值： ``0.5`` 。

    返回：
        Tensor，shape和数据类型与参数 `input` 相同。
