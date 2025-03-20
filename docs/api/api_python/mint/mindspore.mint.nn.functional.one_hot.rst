mindspore.mint.nn.functional.one_hot
======================================

.. py:function:: mindspore.mint.nn.functional.one_hot(tensor, num_classes=-1)

    返回一个one-hot类型的Tensor。

    生成一个新的Tensor，由索引 `tensor` 表示的位置取值为 `1` ，而在其他所有位置取值为 `0` 。

    参数：
        - **tensor** (Tensor) - 输入索引，shape为 :math:`(X_0, \ldots, X_n)` 的Tensor。数据类型必须为int32或int64。维度不能大于7。
        - **num_classes** (int，可选) - 输入的Scalar，定义one-hot的深度，默认值： ``-1`` 。

    返回：
        Tensor，one-hot类型的Tensor。

    异常：
        - **TypeError** - `num_classes` 的数据类型不是int。
        - **TypeError** - `tensor` 的数据类型不是int32或者int64。
        - **ValueError** - `num_classes` 的输入值小于-1。
