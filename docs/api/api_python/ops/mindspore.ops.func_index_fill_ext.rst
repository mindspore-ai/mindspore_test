mindspore.ops.index_fill_ext
=============================

.. py:function:: mindspore.ops.index_fill_ext(input, dim, index, value)

    使用指定的标量或张量值填充给定维度 `dim` 上的 `input` 张量的索引位置。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 要填充的目标张量。`input` 必须是一个Tensor，且其 `dtype` 需与 `value` 相匹配。
        - **dim** (int) - 指定填充的维度，`dim` 必须是一个整数。
        - **index** (Tensor) - 1D 张量，指定要填充的索引位置。`index` 必须为一个整数类型的张量，并且它的值必须在 `input` 张量的合法范围内。
        - **value** (Union(Tensor, Number)) - 用来填充指定索引位置的值。`value` 可以是标量（如数字）或与 `input` 相同类型和shape的张量。

    返回：
        Tensor - 填充后的张量。返回值的形状与 `input` 相同，但在指定索引位置的值已被 `value` 填充。

    异常：
        - **TypeError** - 如果 `dim` 不是整数，或 `input`、`index`、`value` 的类型不符合要求。
        - **AttributeError** - 如果 `value` 为 `float` 类型或其他不支持的类型时，可能引发 `'float' object has no attribute 'dtype'` 错误。
        - **IndexError** - 如果 `index` 中的某个值超出了 `input` 张量在指定维度上的合法索引范围，将抛出索引超出范围的错误。例如，如果 `index` 的值为 `100`，而 `input` 的形状为 `[7]`，则会抛出 "Index value[100] is out of range, it should be smaller than [7]" 错误。
        - **ValueError** - 如果 `dim` 的值超出了 `input` 张量的维度范围，可能会抛出维度相关的错误。
        - **RuntimeError** - 如果输入的张量的维度、形状或索引值不符合预期，可能会出现类似 `aclIndexFillTensorGetWorkspaceSize call failed` 的运行时错误，表示内存分配或索引范围超出了预期。
