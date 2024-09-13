mindspore.numpy.repeat
=================================

.. py:function:: mindspore.numpy.repeat(a, repeats, axis=None)

    重复数组的元素。

    参数：
        - **a** (Tensor) - 输入数组。
        - **repeats** (int, ints的序列) - 每个元素的重复次数。 ``repeats`` 将广播到适应指定轴的shape。
        - **axis** (int, 可选) - 沿着该轴重复元素。如果未指定轴，输入数组将被展平，并返回一个一维的输出数组。默认值： ``None`` 。

    返回：
        Tensor，输出数组，除了指定的轴外，具有与 ``a`` 相同的shape。

    异常：
        - **ValueError** - 如果 ``axis`` 超出范围。
        - **TypeError** - 如果输入 ``a`` 不是Tensor。