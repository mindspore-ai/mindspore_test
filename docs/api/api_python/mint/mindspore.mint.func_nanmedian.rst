mindspore.mint.nanmedian
========================

.. py:function:: mindspore.mint.nanmedian(input)

    计算 `input` 上的中值，并且忽略所有 ``NaN`` 值。

    当 `input` 中不存在 ``NaN`` 值时，此函数的返回与 :func:`mindspore.mint.median` 相同。但如果 `input` 中存在一个或多个
    ``NaN`` 值， :func:`mindspore.mint.median` 总是返回 ``NaN`` 而该函数返回 `input` 中排除了所有 ``NaN`` 元素后的中值。\
    如果 `input` 中的所有元素都是 ``NaN`` 则该函数也将返回一个 ``NaN`` 。

    .. note::
        - 当前Tensor为空时，将返回 ``NaN`` 。如果数据类型不支持 ``NaN`` 则返回 ``0`` 。
        - 此重载在GRAPH_MODE(02)模式下不可用。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 任意维度的Tensor。

    返回：
        Tensor，数据类型与 `input` 保持一致，shape为空，仅包含1个元素，表示输入Tensor排除 ``NaN`` 后的中值。

    .. py:function:: mindspore.mint.nanmedian(input, dim, keepdim=False)
        :noindex:

    输出一个Tuple `(values, indices)` 表示 `input` 按 `dim` 指定的维度上每一行在忽略所有 ``NaN`` 值后的中值与该中值的索引。

    当 `input` 的某一行中不存在 ``NaN`` 值时，此函数对该行的计算结果与 :func:`mindspore.mint.median` 相同。但如果该行中存在\
    一个或多个 ``NaN`` 值， :func:`mindspore.mint.median` 在该行返回 ``NaN`` 而该函数返回 `input` 中排除了所有 ``NaN`` 元素\
    后的中值。如果 `input` 中某一行的所有元素都是 ``NaN`` 则函数也将对该行返回一个 ``NaN`` 。

    两个输出Tensor的shape可以描述为以下形式，其中 `r` 为 `input` 的维度数且 :math:`s_x` 表示 `input` 在第 `x` 维的长度：

    .. math::

        shape = \begin{cases}
        [s_{0}, ... , s_{dim - 1}, s_{dim + 1}, ... , s_{r - 1}]    & \text{if } keepdim = False \\
        [s_{0}, ... , s_{dim - 1}, 1, s_{dim + 1}, ... , s_{r - 1}] & \text{if } keepdim = True \\
        \end{cases}

    .. note::
        - 当 `input` 在被 `dim` 选中的维度长度为0，将引发 ``IndexError`` 。
        - `input` 不允许为标量Tensor。

    .. warning::
        - 如果 `input` 在某一行的中值不唯一（包括全为 ``NaN`` 时），则 `indices` 对该行不一定返回第一个出现的中值的索引。
        - 这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 任意维度的Tensor。
        - **dim** (int) - 指定计算的维度。
        - **keepdim** (bool, 可选) - 是否保留被 `dim` 指定的维度。默认值： ``False`` 。

    返回：
        - **values** (Tensor) - 中值Tensor，数据类型与 `input` 相同，shape见上文的公式。
        - **indices** (Tensor) - `values` 的每一个元素在 `input` 的对应行中的索引，数据类型为int64，shape见上文的公式。

    异常：
        - **IndexError** - `input` 在被 `dim` 选中的维度长度为0。
        - **RuntimeError** - `dim` 的取值不在 `[-input.ndim, input.ndim)` 的范围内。
