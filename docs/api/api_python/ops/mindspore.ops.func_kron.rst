mindspore.ops.kron
==================

.. py:function:: mindspore.ops.kron(input, other)

    计算 `input` 和 `other` 的Kronecker积：:math:`input ⊗ other` 。
    如果 `input` 是shape为 :math:`(r0, r1, ... , rN)` 的Tensor ， `other` 是shape为 :math:`(s0, s1, ... , sN)` 的Tensor ，则计算结果为shape为 :math:`(r0 * s0, r1 * s1, ... , rN * sN)` 的Tensor ，计算公式如下：

    .. math::
        (input ⊗ y)_{k_{0},k_{1},...k_{n}} =
        input_{i_{0},i_{1},...i_{n}} * other_{j_{0},j_{1},...j_{n}},

    其中，对于所有的 0 ≤ `t` ≤ `n`，都有 :math:`k_{t} = i_{t} * b_{t} + j_{t}` 。如果其中一个Tensor维度小于另外一个，
    则对维度较小的Tensor进行unsqueeze补维操作，直到两个Tensor维度相同为止。

    .. note::
        支持实数和复数类型的输入。

    参数：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(r0, r1, ... , rN)` 。
        - **other** (Tensor) - 输入Tensor，shape为 :math:`(s0, s1, ... , sN)` 。

    返回：
        Tensor，shape为 :math:`(r0 * s0, r1 * s1, ... , rN * sN)` 。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `other` 不是Tensor。
