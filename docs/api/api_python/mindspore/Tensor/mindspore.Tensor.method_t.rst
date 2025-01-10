mindspore.Tensor.t
==================

.. py:method:: mindspore.Tensor.t()

    转置 `self` 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    返回：
        Tensor，转置2维Tensor，一维Tensor按原样返回。

    异常：
        - **ValueError** - 输入Tensor维度大于2。
        - **ValueError** - 输入空Tensor。
        - **TypeError** - 输入非Tensor。
