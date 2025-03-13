mindspore.ops.isnan
====================

.. py:function:: mindspore.ops.isnan(input)

    判断输入数据每个位置上的值是否是NaN。

    .. math::

        out_i = \begin{cases}
          & \ True,\ \text{ if } input_{i} = \text{NaN} \\
          & \ False,\ \text{ if } input_{i} \ne  \text{NaN}
        \end{cases}

    其中 :math:`NaN` 表示不是number。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，输出的shape与输入相同，数据类型为bool。

    异常：
        - **TypeError** - `input` 不是Tensor。
