mindspore.mint.mul
===========================

.. py:function:: mindspore.mint.mul(input, other)

    将 `other` 与 `input` 相乘。

    .. math::

        out_{i} = input_{i} \times other_{i}

    .. note::
        - 当两个输入shape不同时，它们必须能够广播到一个共同的shape。
        - 两个输入遵守隐式类型转换规则以使数据类型保持一致。

    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入。是一个 number.Number、
          一个 bool 或一个数据类型为
          `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html>`_ 或
          `bool <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html>`_ 的Tensor。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入。是一个 number.Number、
          一个 bool 或一个数据类型为
          `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html>`_ 或
          `bool <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html>`_ 的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高的类型。

    异常：
        - **TypeError** - 如果 `input`、 `other` 不是以下之一：Tensor、number.Number、bool。
