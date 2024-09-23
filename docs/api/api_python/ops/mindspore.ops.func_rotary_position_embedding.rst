mindspore.ops.rotary_position_embedding
=======================================

.. py:function:: mindspore.ops.rotary_position_embedding(x, cos, sin, mode=0)

    旋转位置编码算法的实现。
    请参阅论文 `Enhanced Transformer with Rotary Position Embedding <https://arxiv.org/pdf/2104.09864.pdf>`_ 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **x** (Tensor) - 4维Tensor，数据类型float16、bfloat16、float32。
        - **cos** (Tensor) - 4维Tensor，数据类型float16、bfloat16、float32。
        - **sin** (Tensor) - 和 `cos` 相同。
        - **mode** (int) - 可选属性，选择计算模式。0表示rotate_half（GPT-NeoX style）， 1表示rotate_interleaved（GPT-J style）。默认值： ``0`` 。

    返回：
        Tensor，其shape和dtype和 `x` 保持一致。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `cos` 不是Tensor。
        - **TypeError** - `sin` 不是Tensor。
        - **TypeError** - `mode` 不是int。
