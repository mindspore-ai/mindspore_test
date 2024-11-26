mindspore.Tensor.ceil
=====================

.. py:method:: mindspore.Tensor.ceil()

    向上取整函数。

    .. math::
        out_i = \lceil self_i \rceil = \lfloor self_i \rfloor + 1

    返回：
        Tensor，shape与 `self` 相同。

    异常：
        - **TypeError** - 如果 `self` 的数据类型不是float16、float32、float64或bfloat16。

          - Ascend: float16、float32、float64、bfloat16。
          - GPU/CPU: float16、float32、float64。
