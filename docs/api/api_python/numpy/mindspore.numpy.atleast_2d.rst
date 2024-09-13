mindspore.numpy.atleast_2d
=================================

.. py:function:: mindspore.numpy.atleast_2d(*arys)

    将输入重新调整为至少二维的数组。

    .. note::
        在图模式下，返回的不是Tensor列表，而是Tensor的tuple。

    参数：
        - **\*arys** (Tensor) - 一个或多个输入Tensor。

    返回：
        Tensor或Tensor列表，每个Tensor满足 :math:`a.ndim >= 2` 。

    异常：
        - **TypeError** - 如果输入不是Tensor。