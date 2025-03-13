mindspore.Tensor.new_full
==========================

.. py:method:: mindspore.Tensor.new_full(size, fill_value, *dtype=None)

    返回一个大小为 `size` 的Tensor，填充值为 `fill_value`。

    参数：
        - **size** (Union[tuple(int), list(int)]) - 定义输出的shape。
        - **fill_value** (Union[number.Number, bool]) - 填充值。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出的数据类型。默认值： ``None`` ，返回的Tensor使用和 `self` 相同的数据类型。

    返回：
        Tensor，shape和dtype由输入定义，填充值为 `fill_value`。

    异常：
        - **TypeError** - 如果 `size` 不是一个个元素为int的Tuple或List。
        - **TypeError** - 如果 `dtype` 不是一个MindSpore的数据类型。
        - **ValueError** - 如果 `size` 包含负数。
