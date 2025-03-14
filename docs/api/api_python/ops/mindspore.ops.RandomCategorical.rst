mindspore.ops.RandomCategorical
===============================

.. py:class:: mindspore.ops.RandomCategorical(dtype=mstype.int64)

    从分类分布中抽取样本。

    .. warning::
        Ascend后端不支持随机数生成结果复现， `seed` 参数将失效。

    参数：
        - **dtype** (mindspore.dtype，可选) - 输出的类型。其值必须是mstype.int16、mstype.int32或mstype.int64。默认值： ``mstype.int64`` 。

    输入：
        - **logits** (Tensor) - 输入Tensor，是一个shape为 :math:`(batch\_size, num\_classes)` 的二维Tensor。
        - **num_sample** (int) - 要抽取的样本数。只允许使用常量值。
        - **seed** (int) - 随机种子值，仅支持常量值。默认值： ``0`` 。

    输出：
        - **output** (Tensor) - 输出Tensor，其shape为 :math:`(batch\_size, num\_samples)` 。

    异常：
        - **TypeError** - 如果 `dtype` 不是mstype.int16、mstype.int32或mstype.int64。
        - **TypeError** - 如果 `logits` 不是Tensor。
        - **TypeError** - 如果 `num_sample` 或者 `seed` 不是int。
