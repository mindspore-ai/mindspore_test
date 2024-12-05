mindspore.mint.dot
====================

.. py:function:: mindspore.mint.dot(input, other)

    计算两个1DTensor的点积。

    参数：
        - **input** (Tensor) - 点积的第一个输入, 须为1D。
        - **other** (Tensor) - 点积的第二个输入，须为1D。

    返回：
        Tensor，shape是[], 类型与input一致。

    异常：
        - **TypeError** - `input` 和 `other` 的数据类型不是tensor。
        - **TypeError** - `input` 或 `other` 的数据类型不是float16, float32, 或bfloat16。
        - **RuntimeError** - `input` 和 `other` 的数据类型不一致。
        - **RuntimeError** - `input` 和 `other` 的shape不一致。
        - **RuntimeError** - `input` 或 `other` 不是1D。