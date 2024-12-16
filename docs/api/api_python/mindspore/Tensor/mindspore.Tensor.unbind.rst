mindspore.Tensor.unbind
========================

.. py:method:: mindspore.Tensor.unbind(dim=0)

    根据指定轴对Tensor进行分解。给定shape为 :math:`(n_1, n_2, ..., n_R)` 的 `self` Tensor，在指定 `dim` 上对其分解，
    则返回多个shape为 :math:`(n_1, n_2, ..., n_{dim}, n_{dim+2}, ..., n_R)` 的Tensor。

    参数：
        - **dim** (int，可选) - 用以分解的维度。取值范围为[-R, R)。默认值为 ``0`` 。

    返回：
        Tensor组成的tuple。每个Tensor对象的shape相同。

    异常：
        - **TypeError** - 如果 `dim` 不是int类型。
        - **ValueError** - 如果 `dim` 超出[-R, R)范围。
