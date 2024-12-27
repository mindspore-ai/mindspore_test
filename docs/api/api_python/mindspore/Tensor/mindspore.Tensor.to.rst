mindspore.Tensor.to
===================

.. py:method:: mindspore.Tensor.to(dtype)

    执行Tensor类型的转换。

    .. note::
        - 如果 `self` 张量已经具有所需的 `mindspore.dtype`，则直接返回 `self`。
          否则，将返回一个具有目标 `mindspore.dtype` 的 `self` 的副本。
        - 当将复数转换为布尔类型时，复数的虚部不被考虑。只要实部非零，它就返回True；否则，返回False。

    参数：
        - **dtype** (dtype.Number) - 输出Tensor的有效数据类型，只允许常量值。

    返回：
        Tensor，其数据类型为 `dtype`。

    异常：
        - **TypeError** -如果 `dtype` 不是数值类型。
