mindspore.Tensor.sub
====================

.. py:method:: mindspore.Tensor.sub(other, *, alpha=1)

    从输入的Tensor中减去另外一个经过缩放的值。

    .. math::

        out_{i} = self_{i} - alpha \times other_{i}

    .. Note::
        - 当两个输入具有不同的shape时，它们的shape必须要能广播为一个共同的shape。
        - 两个输入和alpha遵循隐式类型转换规则，使数据类型保持一致。

    参数：
        - **other** (Union[Tensor, number.Number, bool]) - 另外一个输入，数据类型为 `number.Number`、 `bool`、 `tensor`。
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_.
        - **alpha** (number.Number) - 默认值为 ``1`` 。用于缩放 `other` 的比例因子。

    返回值：
        Tensor，shape与`self` and `other` 广播后的shape相同，并且数据类型是两个输入和alpha之间具有更高精度或更高数字的类型。

    异常：
        - **TypeError** - 如果 `self` 和 `other` 的类型，或者 `alpha` 不是以下类型：`number.Number`、 `bool`、 `Tensor`。
        - **TypeError** - 如果 `alpha` 是浮点类型，但是 `self` 和 `other` 却不是浮点类型。
        - **TypeError** - 如果 `alpha` 是bool类型，但是 `self` 和 `other` 却不是bool类型。

    .. py:method:: mindspore.Tensor.sub(y)

    详情请参考 :func:`mindspore.ops.sub`。
