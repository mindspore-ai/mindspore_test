mindspore.mint.nn.Embedding
===========================

.. py:class:: mindspore.mint.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, dtype=None)

    以 `input` 中的值作为索引，从 `weight` 中查询对应的embedding向量。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 在Ascend后端， `input` 的值非法将导致不可预测的行为。

    参数：
        - **num_embeddings** (int) - embedding词典的大小。
        - **embedding_dim** (int) - 每个embedding向量的长度。
        - **padding_idx** (int, 可选) - 如果值不为 ``None``，则对应的embedding向量不会在训练中更新。且当Embedding层新构建时， `padding_idx` 处embedding向量值将默认置为0。取值范围： `[-num_embeddings, num_embeddings)` 。默认值： ``None`` 。
        - **max_norm** (float, 可选) - 如果给定非 ``None`` 值，则先求出 `input` 指定位置的 `weight` 的p-范数结果（p的值通过 `norm_type` 指定），然后对 `result > max_norm` 位置的 `weight` 进行原地更新，
          更新公式：:math:`\frac{max\_norm}{result+1e^{-7}}` 。默认值： ``None`` 。
        - **norm_type** (float, 可选) - 指定p-范数计算中的p值。默认值： ``2.0`` 。
        - **scale_grad_by_freq** (bool, 可选) - 如果值为 ``True`` ，则反向梯度值会按照 `input` 中索引值重复的次数进行缩放。默认值： ``False`` 。
        - **sparse** (bool, 可选) - 如果值为 ``True`` ， 则 `weight` 为稀疏Tensor，该场景暂未支持。默认值： ``False`` 。
        - **_weight** (Tensor, 可选) - 用于初始化Embedding的权重 `weight` 。如果为 ``None``，则权重将为从正态分布初始化 :math:`{N}(\text{sigma=1.0}, \text{mean=0.0})` 。默认值： ``None`` 。
        - **_freeze** (bool, 可选) - 是否冻结该模型的可学习参数 `weight` 。默认值 ``False`` 。
        - **dtype** (mindspore.dtype, 可选) - 指定Embedding权重 `weight` 的类型。当 `_weight` 不是 ``None`` 时该参数无效。默认值： ``None`` 。

    可变参数：
        - **weight** (Parameter) - 模型的可学习参数，shape为 `(num_embeddings, embedding_dim)`，从正态分布 :math:`{N}(\text{sigma=1.0}, \text{mean=0.0})` 或 `_weight` 初始化。

    输入：
        - **input** (Tensor) - 用于检索embedding向量的索引。数据类型为int32或int64，取值范围： `[0, num_embeddings)` 。

    输出：
        Tensor，数据类型与 `weight` 保持一致，shape为 :math:`(*input.shape, embedding\_dim)` 。

    异常：
        - **TypeError** - `num_embeddings` 不是int。
        - **TypeError** - `embedding_dim` 不是int。
        - **ValueError** - `padding_idx` 的取值不在有效范围。
        - **TypeError** - `max_norm` 不是float。
        - **TypeError** - `norm_type` 不是float。
        - **TypeError** - `scale_grad_by_freq` 不是bool型。
        - **ValueError** - `weight.shape` 非法。
        - **TypeError** - `dtype` 不是mindspore.dtype。