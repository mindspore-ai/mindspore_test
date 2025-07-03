mindspore.ops.hstack
====================

.. py:function:: mindspore.ops.hstack(tensors)

    将多个tensor沿着水平方向堆叠。

    .. note::
        - 在 `graph mode <https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html>`_ 下，不支持 ``mindspore.float64`` 类型的8-D tensor的动态rank输入。
        - 对于一维tensor，沿第一个轴堆叠。其他维度的tensor沿第二个轴堆叠。
        - 对于维度大于一的tensor，除了第二个轴外，所有tensor的shape必须相同。对于一维tensor，可拥有任意的长度。

    参数：
        - **tensors** (Union[tuple[Tensor], list[Tensor]]) - 输入tensors。

    返回：
        Tensor
