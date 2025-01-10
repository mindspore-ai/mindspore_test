mindspore.Tensor.sinc
=====================

.. py:method:: mindspore.Tensor.sinc()

    计算输入的归一化正弦值。

    .. math::
        out_i = \begin{cases} \frac{sin(\pi self_i)}{\pi self_i} & self_i\neq 0\\ 
        1 & self_i=0 \end{cases}

    返回：
        Tensor，shape与 `self` 相同。
        当输入类型为int或bool时，返回值类型为float32。
        否则，返回值类型与输入类型相同。

    异常：
        - **TypeError** - 如果 `self` 不是Tensor。
