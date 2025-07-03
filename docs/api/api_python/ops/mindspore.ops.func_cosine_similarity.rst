mindspore.ops.cosine_similarity
================================

.. py:function:: mindspore.ops.cosine_similarity(x1, x2, dim=1, eps=1e-08)

    沿指定维度计算两个输入tensor之间的余弦相似度。

    .. note::
        当前暂不支持对输入进行广播。

    .. math::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    参数：
        - **x1** (Tensor) - 第一个输入tensor。
        - **x2** (Tensor) - 第二个输入tensor。
        - **dim** (int, 可选) - 指定维度。默认 ``1`` 。
        - **eps** (float, 可选) - 极小值，用于避免除零的情况。默认 ``1e-08`` 。

    返回：
        Tensor
