mindspore.numpy.logical_not
=================================

.. py:function:: mindspore.numpy.logical_not(a, dtype=None)

    逐元素计算 ``a`` 的逻辑非（NOT）的真值。

    .. note::
        不支持Numpy的 ``out`` 、 ``where`` 、 ``casting`` 、 ``order`` 、 ``subok`` 、 ``signature`` 和 ``extobj`` 参数。

    参数：
        - **a** (Tensor) - 输入Tensor，其dtype为bool。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor或标量。对 ``a`` 中的元素执行逻辑非（NOT）操作的Boolean结果，具有与 ``a`` 相同的shape。如果 ``a`` 是标量，则返回标量。

    异常：
        - **TypeError** - 如果输入不是Tensor或其dtype不是bool。