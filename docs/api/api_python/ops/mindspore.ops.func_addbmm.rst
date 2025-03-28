﻿mindspore.ops.addbmm
=====================

.. py:function:: mindspore.ops.addbmm(input, batch1, batch2, *, beta=1, alpha=1)

    对 `batch1` 和 `batch2` 中的矩阵相乘，按照第一维度累计相加，再与 `input` 相加。

    .. note::
        - `batch1` 和 `batch2` 必须是三维的tensor，且包含相同数量的矩阵。
        - 如果 `batch1` 是大小为 :math:`(C, W, T)` 的tensor，  `batch2` 是大小为 :math:`(C, T, H)` 的tensor，
          则 `input` 必须能够与大小为 :math:`(W, H)` 的tensor进行广播，且输出将是大小为 :math:`(W, H)` 的tensor。
        - 若 `beta` 为0，那么 `input` 将会被忽略。

    .. math::
        output = \beta input + \alpha (\sum_{i=0}^{b-1} {batch1_i @ batch2_i})

    参数：
        - **input** (Tensor) - 输入tensor。
        - **batch1** (Tensor) - 第一个将被乘batch矩阵。
        - **batch2** (Tensor) - 第二个将被乘batch矩阵。

    关键字参数：
        - **beta** (Union[int, float]，可选) - `input` 的尺度因子。默认 ``1`` 。
        - **alpha** (Union[int, float]，可选) - （ `batch1` @ `batch2` ）的尺度因子。默认 ``1`` 。

    返回：
        Tensor
