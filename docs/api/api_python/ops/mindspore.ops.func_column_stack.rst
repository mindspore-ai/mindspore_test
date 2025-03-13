mindspore.ops.column_stack
==========================

.. py:function:: mindspore.ops.column_stack(tensors)

    将多个Tensor沿着水平方向维度堆叠成一个Tensor，即按列拼接。Tensor的其他维度在拼接后保持不变。\
    类似于 :func:`mindspore.ops.hstack`。

    参数：
        - **tensors** (Union[tuple[Tensor], list[Tensor]]) - 包含多个Tensor的tuple或list。除了需要拼接的轴外，所有\
          Tensor必须具有相同的shape。

    返回：
        Tensor，将输入Tensor堆叠后的结果。

    异常：
        - **TypeError** - 如果 `tensors` 不是list或tuple。
        - **TypeError** - 如果 `tensors` 的元素不是Tensor。
        - **ValueError** - 如果 `tensors` 为空。
