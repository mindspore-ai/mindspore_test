mindspore.ops.standard_laplace
==============================

.. py:function:: mindspore.ops.standard_laplace(shape, seed=None)

    根据标准Laplace（mean=0, lambda=1）分布生成随机数。

    .. math::
        \text{f}(x) = \frac{1}{2}\exp(-|x|)

    .. warning::
        Ascend后端不支持随机数重现功能， `seed` 参数不起作用。

    参数：
        - **shape** (Union[tuple, Tensor]) - 指定生成随机数的shape。
        - **seed** (int, 可选) - 随机数种子。默认 ``None`` 。

    返回：
        Tensor

    异常：
        - **ValueError** - `shape` 为tuple时，包含非正的元素。
        - **ValueError** - `shape` 为秩不等于1的tensor。
