mindspore.numpy.atleast_1d
=================================

.. py:function:: mindspore.numpy.atleast_1d(*arys)

    将输入转换为至少一维的数组。
    标量输入将被转换为一维数组，而高维输入则保持不变。

    .. note::
        在图模式下，返回的不是Tensor列表，而是Tensor的tuple。

    参数：
        - **\*arys** (Tensor) - 一个或多个输入Tensor。

    返回：
        Tensor或Tensor列表，每个Tensor满足 :math:`a.ndim >= 1` 。

    异常：
        - **TypeError** - 如果输入不是Tensor。