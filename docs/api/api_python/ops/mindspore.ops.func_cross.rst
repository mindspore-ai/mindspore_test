mindspore.ops.cross
====================

.. py:function:: mindspore.ops.cross(input, other, dim=None)

    按指定维度计算两个输入tensor的叉积。

    .. note::
        `input` 和 `other` 必须有相同的shape，且指定的 `dim` 上size必须为3。 如果不指定 `dim` ，则使用第一个size为3的维度。

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **other** (Tensor) - 第二个输入tensor。
        - **dim** (int，可选) - 指定维度。默认 ``None`` 。

    返回：
        Tensor
