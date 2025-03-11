mindspore.ops.hstack
====================

.. py:function:: mindspore.ops.hstack(tensors)

    将多个Tensor沿着水平方向进行堆叠。
    对于1-D Tensor，沿第一个轴进行堆叠。其他维度的Tensor沿第二个轴进行堆叠。

    .. note::
        Float64类型的8-D Tensor的动态rank输入不支持在 `图模式(mode=mindspore.GRAPH_MODE) <https://www.mindspore.cn/docs/zh-CN/master/model_train/program_form/static_graph.html>`_ 下执行。

    参数：
        - **tensors** (Union[tuple[Tensor], list[Tensor]]) - 包含多个Tensor。对于维度大于1-D的Tensor，除了第二个轴外，所有的\
          Tensor必须有相同的shape。对于1-D Tensor，可拥有任意的长度。

    返回：
        堆叠后的Tensor。

    异常：
        - **TypeError** - `tensors` 不是 list或tuple。
        - **TypeError** - `tensors` 的元素不是 Tensor。
        - **ValueError** - `tensors` 为空。
