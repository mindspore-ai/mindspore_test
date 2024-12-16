mindspore.Tensor.bincount
=========================

.. py:method:: mindspore.Tensor.bincount(weights=None, minlength=0)

    统计 self 中每个值的出现次数。

    如果不指定 minlength ，输出Tensor的长度为输入 self 中最大值加1。如果指定了 minlength，则输出Tensor的长度为 self 中最大值加1和 minlength 的最大值。

    输出Tensor中每个值标记了该索引值在 self 中的出现次数。如果指定了 weights，对输出的结果进行加权处理，即 :math:`out[n]+=weight[i]` 而不是 :math:`out[n]+=1`。

    参数：
        - **weights** (Tensor，可选) - 权重，与 self shape相同。默认值： ``None`` 。
        - **minlength** (int，可选) - 输出Tensor的最小长度，应为非负数。默认值： ``0`` 。

    返回：
        Tensor，如果输入为非空，输出shape为 :math:`(max(max(self)+1, minlength), )`，否则shape为 :math:`(0, )`。

    异常：
        - **TypeError** - 如果 `weights` 不是Tensor。
        - **ValueError** - 如果 `self` 中存在负数。
        - **ValueError** - 如果 `self` 不是一维的，或者 `self` 和 `weights` 不具有相同的shape。
