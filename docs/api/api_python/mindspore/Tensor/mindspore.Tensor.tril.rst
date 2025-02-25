mindspore.Tensor.tril
=====================

.. py:method:: mindspore.Tensor.tril(diagonal=0)

    返回输入Tensor的下三角形部分(包含对角线和下面的元素)，并将其他元素设置为0。

    参数：
        - **diagonal** (int，可选) - 指定对角线位置，默认值： ``0`` ，指定主对角线。

    返回：
        Tensor。其数据类型和shape维度与输入相同。

    异常：
        - **TypeError** - 如果 `diagonal` 不是int。
        - **TypeError** - 如果输入Tensor的数据类型既不是number也不是bool。
        - **ValueError** - 如果输入Tensor的秩小于2。
