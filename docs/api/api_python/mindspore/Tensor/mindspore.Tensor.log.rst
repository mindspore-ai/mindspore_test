mindspore.Tensor.log
====================

.. py:method:: mindspore.Tensor.log()

    逐元素返回Tensor的自然对数。

    .. math::
        y_i = \log_e(self_i)

    .. warning::
        如果输入值在(0, 0.01]或[0.95, 1.05]范围内，则输出精度可能会存在误差。

    .. note::
        `self` 的值必须大于0。

    返回：
        Tensor，具有与 `self` 相同的shape。
