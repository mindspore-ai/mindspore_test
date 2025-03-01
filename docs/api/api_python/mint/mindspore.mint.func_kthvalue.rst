mindspore.mint.kthvalue
=======================

.. py:function:: mindspore.mint.kthvalue(input, k, dim=-1, keepdim=False)

    计算输入Tensor沿指定维度 `dim` 的第 `k` 小值，并返回一个由（ `values`, `indices` ）组成的Tuple，其中 `values` 包含第 `k` 小的元素， `indices` 提供每个对应元素的索引。

    参数：
        - **input** (Tensor) - 输入Tensor，可以是任意维度。设输入Tensor的shape为：:math:`(input_1, input_2, ..., input_N)` 。
        - **k** (int) - 指定要检索的第k小的元素。
        - **dim** (int，可选) - 指定用于查找第k小值的维度。默认值为 ``-1`` 。
        - **keepdim** (bool，可选) - 是否保留维度。如果设置为 ``True`` ，输出将与输入保持相同的维度；如果设置为 ``False`` ，输出将减少对应的维度。默认值为 ``False`` 。

    返回：
        返回一个包含 `values` 和 `indices` 的Tuple。

        - **values** (Tensor) - 输入Tensor的第k小值，与输入Tensor的数据类型相同。

          - 如果 `keepdim` 为 ``True`` ，输出Tensor的shape为：:math:`(input_1, input_2, ..., input_{dim-1}, 1, input_{dim+1}, ..., input_N)`。
          - 如果 `keepdim` 为 ``False`` ，输出Tensor的shape为：:math:`(input_1, input_2, ..., input_{dim-1}, input_{dim+1}, ..., input_N)`。

        - **indices** (Tensor) - 输入Tensor第k小值的索引，其shape与 `values` 相同，数据类型为int64。

    异常：
        - **TypeError** - 如果 `k` 或者 `dim` 不是int。
        - **TypeError** - 如果 `keepdim` 不是bool。
        - **TypeError** - 如果 `input` 的数据类型不支持。
        - **ValueError** - 如果 `input` 为空Tensor。
        - **RuntimeError** - 如果 `k` 不在有效范围内。
