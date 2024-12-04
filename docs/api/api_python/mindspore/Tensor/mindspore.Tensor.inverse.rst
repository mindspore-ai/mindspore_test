mindspore.Tensor.inverse
========================

.. py:method:: mindspore.Tensor.inverse()

    计算输入矩阵的逆。

    返回：
        Tensor，其类型和shape与 `self` 的类型和shape相同。

    异常：
        - **ValueError** - `self` 最后两个维度的大小不相同。
        - **ValueError** - `self` 的维数小于2。

