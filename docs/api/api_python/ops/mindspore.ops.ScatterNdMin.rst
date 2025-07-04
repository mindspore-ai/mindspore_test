mindspore.ops.ScatterNdMin
===========================

.. py:class:: mindspore.ops.ScatterNdMin(use_locking=False)

    对Tensor中的单个值或切片应用稀疏最小值。

    使用给定值通过最小值运算和输入索引更新 `input_x` 的值。为便于使用更新后的值，函数返回 `input_x` 的副本。

    更多参考详见 :func:`mindspore.ops.scatter_nd_min` 。

    参数：
        - **use_locking** (bool，可选) - 是否启用锁保护。默认值： ``False`` 。

    输入：
        - **input_x** (Union[Parameter, Tensor]) - 输入参数，数据类型为Parameter或Tensor。
        - **indices** (Tensor) - 指定最小值操作的索引，数据类型为mindspore.int32或mindspore.int64。索引的rank必须至少为2，并且 `indices.shape[-1] <= len(shape)` 。
        - **updates** (Tensor) - 指定与 `input_x` 操作的Tensor，数据类型与 `input_x` 相同，shape为 `indices.shape[:-1] + x.shape[indices.shape[-1]:]` 。

    输出：
        Tensor，更新后的 `input_x` ，shape和数据类型与 `input_x` 相同。
