mindspore.Tensor.short
=======================

.. py:method:: mindspore.Tensor.short()

    将输入Tensor转换为 `int16` 类型并返回其副本，与 `self.astype(mstype.int16)` 等价，当Tensor中的值为浮点数时，则会丢弃小数部分，具体请参考 :func:`mindspore.Tensor.astype`。

    返回：
        Tensor，其数据类型为 `int16`。
