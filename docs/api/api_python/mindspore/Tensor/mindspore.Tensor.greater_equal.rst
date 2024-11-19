mindspore.Tensor.greater_equal
==============================

.. py:method:: Tensor.greater_equal(other)

    逐元素比较 'self' Tensor 是否大于等于第二个Tensor。

    更多参考详见 :func:`mindspore.ops.ge`。

    参数：
        - **other** (Union[Tensor, Number]) - 该输入应该是一个Number或数据类型为number或bool_的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为bool。
