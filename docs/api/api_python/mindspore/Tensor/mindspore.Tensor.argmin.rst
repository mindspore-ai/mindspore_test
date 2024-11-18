mindspore.Tensor.argmin
=======================

.. py:method:: mindspore.Tensor.argmin(axis=None, keepdims=False)

    返回 `self` 在指定轴上的最小值索引。

    如果 `self` 的shape为 :math:`(self_1, ..., self_N)` ，则输出Tensor的shape为 :math:`(self_1, ..., self_{axis-1}, self_{axis+1}, ..., self_N)` 。

    参数：
        - **axis** (Union[int, None]，可选) - 指定计算轴。如果是 ``None`` ，将会返回扁平化Tensor在指定轴上的最小值索引。默认值： ``None`` 。
        - **keepdims** (bool，可选) - 输出Tensor是否保留指定轴。如果 `axis` 是 ``None`` ，忽略该选项。默认值： ``False`` 。

    返回：
        Tensor，输出为指定轴上 `self` 最小值的索引。

    异常：
        - **TypeError** - `axis` 不是int。

    .. py:method:: mindspore.Tensor.argmin(dim=None, keepdim=False)
        :noindex:

    返回 `self` 在指定轴上的最小值索引。

    如果 `self` 的shape为 :math:`(self_1, ..., self_N)` ，则输出Tensor的shape为 :math:`(self_1, ..., self_{axis-1}, self_{axis+1}, ..., self_N)` 。

    参数：
        - **dim** (Union[int, None]，可选) - 指定计算维度。如果是 ``None`` ，将会返回扁平化Tensor在指定维度上的最小值索引。默认值： ``None`` 。
        - **keepdim** (bool，可选) - 输出Tensor是否保留指定轴。如果 `dim` 是 ``None`` ，忽略该选项。默认值： ``False`` 。

    返回：
        Tensor，输出为指定维度上 `self` 最小值的索引。

    异常：
        - **TypeError** - `dim` 不是int。
