mindspore.Tensor.triu
=====================

.. py:method:: Tensor.triu(diagonal=0)

    返回 `self` 的上三角形部分(包含对角线和下面的元素)，并将其他元素设置为0。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **diagonal** (int，可选) - 指定对角线位置，默认值： ``0`` ，指定主对角线。

    返回：
        Tensor，其数据类型和shape与 `self` 相同。

    异常：
        - **TypeError** - 如果 `diagonal` 不是int。
        - **ValueError** - 如果 `self` 的维度小于2。
