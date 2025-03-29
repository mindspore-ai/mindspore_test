mindspore.Tensor.nanmedian
===========================

.. py:method:: mindspore.Tensor.nanmedian()

    计算当前Tensor上的中值，并且忽略所有 ``NaN`` 值。

    在当前Tensor中不存在 ``NaN`` 值时，此函数的返回与 :func:`mindspore.Tensor.median` 相同。但如果当前Tensor中存在一个或多个
    ``NaN`` 值， :func:`mindspore.Tensor.median` 总是返回 ``NaN`` 而该函数返回当前Tensor中排除了所有 ``NaN`` 元素后的中值。\
    如果当前Tensor中的所有元素都是 ``NaN`` 则该函数也将返回一个 ``NaN`` 。

    .. note::
        - 当前Tensor为空时，将返回 ``NaN`` 。如果数据类型不支持 ``NaN`` 则返回 ``0`` 。
        - 此重载在GRAPH_MODE(02)模式下不可用。
        - 此重载在GRAPH_MODE下仅支持 ``Ascend`` 平台。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    返回：
        Tensor，数据类型与当前Tensor保持一致，shape为空，仅包含1个元素，表示当前Tensor排除 ``NaN`` 后的中值。

    .. py:method:: mindspore.Tensor.nanmedian(dim, keepdim=False)
        :noindex:

    输出一个Tuple `(values, indices)` 表示当前Tensor按 `dim` 指定的维度上每一行在忽略所有 ``NaN`` 值后的中值与该中值的索引。

    在当前Tensor的某一行中不存在 ``NaN`` 值时，此函数对该行的计算结果与 :func:`mindspore.Tensor.median` 相同。但如果该行中存在\
    一个或多个 ``NaN`` 值， :func:`mindspore.Tensor.median` 在该行返回 ``NaN`` 而该函数返回当前Tensor中排除了所有 ``NaN`` 元素\
    后的中值。如果当前Tensor中某一行的所有元素都是 ``NaN`` 则函数也将对该行返回一个 ``NaN`` 。

    两个输出Tensor的shape可以描述为以下形式，其中 `r` 为当前Tensor的维度数且 :math:`s_x` 表示当前Tensor在第 `x` 维的长度：

    .. math::

        shape = \begin{cases}
        [s_{0}, ... , s_{dim - 1}, s_{dim + 1}, ... , s_{r - 1}]    & \text{if } keepdim = False \\
        [s_{0}, ... , s_{dim - 1}, 1, s_{dim + 1}, ... , s_{r - 1}] & \text{if } keepdim = True \\
        \end{cases}

    .. note::
        - 在当前Tensor在被 `dim` 选中的维度长度为0，其行为取决于平台：

          - 在 ``Ascend`` 平台上将引发 ``IndexError`` 。
          - 在 ``CPU`` 平台将仅产生WARNING日志。

        - 在 ``Ascend`` 平台上，当前Tensor不允许为标量Tensor。
        - 此重载在GRAPH_MODE下仅支持 ``CPU`` 平台。

    .. warning::
        - 如果当前Tensor在某一行的中值不唯一（包括全为 ``NaN`` 时），则 `indices` 对该行不一定返回第一个出现的中值的索引。
        - 这是一个实验性API，后续可能修改或删除。

    参数：
        - **dim** (int) - 指定计算的维度。
        - **keepdim** (bool, 可选) - 是否保留被 `dim` 指定的维度。默认值： ``False`` 。

    返回：
        - **values** (Tensor) - 中值Tensor，数据类型与当前Tensor相同，shape见上文的公式。
        - **indices** (Tensor) - `values` 的每一个元素在当前Tensor的对应行中的索引，数据类型为int64，shape见上文的公式。

    异常：
        - **IndexError** - 在 ``Ascend`` 平台上，当前Tensor在被 `dim` 选中的维度长度为0。
        - **ValueError** - 在 ``CPU`` 平台上， `dim` 的取值不在 `[-self.ndim, self.ndim)` 的范围内。
        - **RuntimeError** - 在 ``Ascend`` 平台上， `dim` 的取值不在 `[-self.ndim, self.ndim)` 的范围内。
