mindspore.ops.dstack
====================

.. py:function:: mindspore.ops.dstack(tensors)

    将多个tensor沿着第三维度进行堆叠。

    .. note::
        - 一维tensor :math:`(N,)` 重新排列为 :math:`(1,N,1)` ，二维tensor :math:`(M,N)` 重新排列为 :math:`(M,N,1)` 。
        - 除了第三个轴外，所有的tensor必须有相同的shape。如果是1-D或2-D的tensor，则它们的shape必须相同。

    参数：
        - **tensors** (Union(List[Tensor], Tuple[Tensor])) - 由多个tensor组成的list或tuple。

    返回：
        Tensor
