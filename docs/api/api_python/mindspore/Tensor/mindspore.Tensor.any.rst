mindspore.Tensor.any
====================

.. py:method:: mindspore.Tensor.any(axis=None, keep_dims=False)

    默认情况下，通过对维度中所有元素进行“逻辑或”来减少 `self` 的维度。也可以沿 `axis` 减少 `self` 的维度。通过控制 `keep_dims` 来确定输出和输入的维度是否相同。

    .. note::
        Tensor类型的 `axis` 仅用作兼容旧版本，不推荐使用。

    参数：
        - **axis** (Union[int, tuple(int), list(int), Tensor], 可选) - 要减少的维度。只允许常量值。假设 `self` 的秩为r，取值范围[-r,r)。默认值： ``None`` ，缩小所有维度。
        - **keep_dims** (bool, 可选) - 如果为 ``True`` ，则保留缩小的维度，大小为1。否则移除维度。默认值： ``False`` 。

    返回：
        Tensor，数据类型是bool。

        - 如果 `axis` 为 ``None`` ，且 `keep_dims` 为 ``False`` ，则输出一个零维Tensor，表示 `self` 中所有元素进行“逻辑或”。
        - 如果 `axis` 为int，例如取值为2，并且 `keep_dims` 为 ``False`` ，则输出的shape为 :math:`(self_1, self_3, ..., self_R)` 。
        - 如果 `axis` 为tuple(int)或list(int)，例如取值为(2, 3)，并且 `keep_dims` 为 ``False`` ，则输出Tensor的shape为 :math:`(self_1, self_4, ..., self_R)` 。
        - 如果 `axis` 为一维Tensor，例如取值为[2, 3]，并且 `keep_dims` 为 ``False`` ，则输出Tensor的shape为 :math:`(self_1, self_4, ..., self_R)` 。

    异常：
        - **TypeError** - `keep_dims` 不是bool类型。
        - **TypeError** - `axis` 不是以下数据类型之一：int、tuple、list或Tensor。

    .. py:method:: mindspore.Tensor.any(dim=None, keepdim=False)
        :noindex:

    默认情况下，通过对维度中所有元素进行“逻辑或”来减少 `self` 的维度。也可以沿 `dim` 减少 `self` 的维度。通过控制 `keepdim` 来确定输出和输入的维度是否相同。

    .. note::
        Tensor类型的 `dim` 仅用作兼容旧版本，不推荐使用。

    参数：
        - **dim** (Union[int, tuple(int), list(int), Tensor], 可选) - 要减少的维度。只允许常量值。假设 `self` 的秩为r，取值范围[-r,r)。默认值： ``None`` ，缩小所有维度。
        - **keepdim** (bool, 可选) - 如果为 ``True`` ，则保留缩小的维度，大小为1。否则移除维度。默认值： ``False`` 。

    返回：
        Tensor，数据类型是bool。

        - 如果 `dim` 为 ``None`` ，且 `keepdim` 为 ``False`` ，则输出一个零维Tensor，表示 `self` 中所有元素进行“逻辑或”。
        - 如果 `dim` 为int，例如取值为2，并且 `keepdim` 为 ``False`` ，则输出的shape为 :math:`(self_1, self_3, ..., self_R)` 。
        - 如果 `dim` 为tuple(int)或list(int)，例如取值为(2, 3)，并且 `keepdim` 为 ``False`` ，则输出Tensor的shape为 :math:`(self_1, self_4, ..., self_R)` 。
        - 如果 `dim` 为一维Tensor，例如取值为[2, 3]，并且 `keepdim` 为 ``False`` ，则输出Tensor的shape为 :math:`(self_1, self_4, ..., self_R)` 。

    异常：
        - **TypeError** - `keepdim` 不是bool类型。
        - **TypeError** - `dim` 不是以下数据类型之一：int、tuple、list或Tensor。
