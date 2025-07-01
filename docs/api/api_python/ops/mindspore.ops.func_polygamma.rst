mindspore.ops.polygamma
=======================

.. py:function:: mindspore.ops.polygamma(n, input)

    计算 `input` 的多伽马函数的 :math:`n` 阶导数。

    .. math::
        \psi^{(a)}(x) = \frac{d^{(a)}}{dx^{(a)}} \psi(x)

    其中 :math:`\psi(x)` 为digamma函数。

    参数：
        - **n** (Tensor) - 多伽马函数求导的阶数。
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor。
