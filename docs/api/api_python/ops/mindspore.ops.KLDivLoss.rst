mindspore.ops.KLDivLoss
=======================

.. py:class:: mindspore.ops.KLDivLoss(reduction='mean')

    计算输入 `logits` 和 `labels` 的KL散度。

    对于相同形状的张量 :math:`x` 和 :math:`target` ，KLDivLoss的计算公式如下：

    .. math::
        L(x, target) = target \cdot (\log target - x)

    输出

    .. math::
        \ell(x, target) = \begin{cases}
        L(x, target), & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L(x, target)), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L(x, target)) / x.\operatorname{shape}[0], & \text{if reduction} = \text{'batchmean';}\\
        \operatorname{sum}(L(x, target)),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    其中 :math:`x` 代表 `logits` ；
    :math:`target` 代表 `labels` ；
    :math:`\ell(x, target)` 为 `output` 。

    .. note::
        - 目前Ascend平台不支持float64数据类型。
        - 仅当 `reduction` 设置为 ``'batchmean'`` 时输出才与Kullback-Leibler散度的数学定义一致。
        - 在Ascend平台上， `reduction` 的可选值为 ``'batchmean'`` 、 ``'none'`` 或 ``'sum'`` 。
        - 在GPU平台上， `reduction` 的可选值为 ``'mean'`` 、 ``'none'`` 或 ``'sum'`` 。
        - 在CPU平台上， `reduction` 的可选值为 ``'mean'`` 、 ``'batchmean'`` 、 ``'none'`` 或 ``'sum'`` 。

    参数：
        - **reduction** (str，可选) - 指定输出结果的计算方式。默认值:  ``'mean'`` 。

          - ``'none'``：不应用规约方法。
          - ``'mean'``：计算输出元素的平均值。
          - ``'sum'``：计算输出元素的总和。
          - ``'batchmean'``：计算批次的平均损失，类似于mean模式。

    输入：
        - **logits** (Tensor) - 数据类型支持float16、float32和float64。
        - **labels** (Tensor) - 标签Tensor，与 `logits` 的shape和数据类型相同。

    输出：
        Tensor或标量。如果 `reduction` 为 ``"none"`` ，则输出为Tensor且与 `logits` 的shape相同，否则为标量。

    异常：
        - **TypeError** - `reduction` 不是str。
        - **TypeError** - `logits` 或 `labels` 不是Tensor。
        - **TypeError** - `logits` 或 `labels` 的数据类型不是支持的类型。
        - **ValueError** - `logits` 和 `labels` 的shape不一致。
        - **RuntimeError** - `logits` 或 `labels` 是标量并且 `reduction` 是 ``'batchmean'`` 。
