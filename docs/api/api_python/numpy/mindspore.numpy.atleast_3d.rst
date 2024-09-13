mindspore.numpy.atleast_3d
=================================

.. py:function:: mindspore.numpy.atleast_3d(*arys)

    将输入重新调整为至少三维的数组。

    .. note::
        在图模式下，返回的不是Tensor列表，而是Tensor的tuple。

    参数：
        - **\*arys** (Tensor) - 一个或多个输入Tensor。

    返回：
        Tensor或Tensor列表，每个Tensor满足 :math:`a.ndim >= 3` 。例如，一个shape为 ``(N,)`` 的一维数组会变为shape为 ``(1, N, 1)`` 的Tensor，一个shape为 ``(M, N)`` 的二维数组会变为shape为 ``(M, N, 1)`` 的Tensor。

    异常：
        - **TypeError** - 如果输入不是Tensor。