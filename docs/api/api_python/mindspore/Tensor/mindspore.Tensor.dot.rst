mindspore.Tensor.dot
====================

.. py:method:: mindspore.Tensor.dot(other)

    计算两个1DTensor的点积。

    参数：
        - **other** (Tensor) - 点积的输入，须为1D。

    返回：
        Tensor，shape是[]，类型与 `self` 一致。

    异常：
        - **TypeError** - `other` 的数据类型不是tensor。
        - **RuntimeError** - `self` 和 `other` 的数据类型不一致。
        - **RuntimeError** - `self` 和 `other` 的shape不一致。
        - **RuntimeError** - `self` 或 `other` 不是1D。