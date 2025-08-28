mindspore.mint.transpose
==========================

.. py:function:: mindspore.mint.transpose(input, dim0, dim1)

    交换Tensor的两个维度。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dim0** (int) - 第一个维度。
        - **dim1** (int) - 第二个维度。

    返回：
        转化后的Tensor，与输入具有相同的数据类型。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `dim0` 或者 `dim1` 不是整数。
        - **ValueError** - 如果 `dim0` 或者 `dim1` 不在 :math:`[-ndim, ndim -1]` 范围内。
