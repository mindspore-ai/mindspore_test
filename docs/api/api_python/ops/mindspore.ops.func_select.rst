mindspore.ops.select
====================

.. py:function:: mindspore.ops.select(condition, input, other)

    根据条件判断tensor中的元素的值，来决定输出中的相应元素是从 `input` （如果元素值为True）还是从 `other` （如果元素值为False）中选择。

    该算法可以被定义为：

    .. math::

        out_i = \begin{cases}
        input_i, & \text{if } condition_i \\
        other_i, & \text{otherwise}
        \end{cases}

    参数：
        - **condition** (Tensor[bool]) - 条件tensor。
        - **input** (Union[Tensor, int, float]) - 第一个被选择的tensor或数字。
        - **other** (Union[Tensor, int, float]) - 第二个被选择的tensor或数字。

    返回：
        Tensor
