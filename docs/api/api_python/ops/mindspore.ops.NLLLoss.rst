mindspore.ops.NLLLoss
======================

.. py:class:: mindspore.ops.NLLLoss(reduction="mean", ignore_index=-100)

    获取预测值和目标值之间的负对数似然损失。

    :math:`reduction = none` 时，负对数似然损失如下：

    .. math::
        \ell(x, t)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top},
        \quad l_{n}=-w_{t_{n}} x_{n, t_{n}},
        \quad w_{c}=\text { weight }[c] \cdot 1

    其中， :math:`x` 表示预测值， :math:`t` 表示目标值， :math:`w` 表示权重， :math:`N` 表示batch size， :math:`c` 限定范围为[0, C-1]，表示类索引，其中 :math:`C` 表示类的数量。

    :math:`reduction \neq none` 时（默认为 ``"mean"`` ），则

    .. math::
        \ell(x, t)=\left[\{\begin{array}{ll}
        \sum_{n=1}^{N} \frac{1}{\sum_{n=1}^{N} w_{t n}} l_{n}, & \text { if reduction }=\text { 'mean'; } \\
        \sum_{n=1}^{N} l_{n}, & \text { if reduction }=\text { 'sum' }
        \end{array}\right]

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的加权平均值。
          - ``"sum"``：计算输出元素的总和。

        - **ignore_index** (int，可选) - 指定标签中需要忽略的值（一般为填充值），使其不对梯度产生影响。默认值： ``-100`` 。

    输入：
        - **logits** (Tensor) - 输入预测值，shape为 :math:`(N, C)` 。数据类型仅支持float32或float16或bfloat16（仅Atlas A2训练系列产品支持）。
        - **labels** (Tensor) - 输入目标值，shape为 :math:`(N,)` ，取值范围为 :math:`[0, C-1]` 。数据类型仅支持uint8或int32或int64。
        - **weight** (Tensor) - 指定各类别的权重。shape为 :math:`(C,)`。数据类型仅支持float32或float16或bfloat16（仅Atlas A2训练系列产品支持）。要求与 `logits` 的数据类型保持一致。

    返回：
        由 `loss` 和 `total_weight` 组成的2个Tensor的tuple。

        - **loss** (Tensor) - 当 `reduction` 为 ``"none"`` 且 `logits` 为二维Tensor时， `loss` 的shape为 :math:`(N,)` ，否则 `loss` 为scalar。 `loss` 的数据类型与 `logits` 相同。
        - **total_weight** (Tensor) - `total_weight` 是scalar，数据类型与 `weight` 相同。
