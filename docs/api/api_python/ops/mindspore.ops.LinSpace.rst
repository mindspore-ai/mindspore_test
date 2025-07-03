mindspore.ops.LinSpace
======================

.. py:class:: mindspore.ops.LinSpace

    创建一个一维Tensor，其包含steps个元素，这些元素在区间[start, end]内均匀分布。

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [start, start+step, start+2*step, ... , end]
        \end{aligned}

    输入：
        - **start** (Tensor) - 区间的起始值。
        - **stop** (Tensor) - 区间的末尾值。
        - **num** (Union[int, Tensor]) - 元素数量。

    输出：
        Tensor
