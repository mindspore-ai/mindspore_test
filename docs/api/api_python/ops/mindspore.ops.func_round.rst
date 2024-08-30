mindspore.ops.round
====================

.. py:function:: mindspore.ops.round(input, *, decimals=None)

    对输入数据进行四舍五入到最接近的整数数值。

    .. math::
        out_i \approx input_i

    .. note::
        Ascend平台支持的输入数据类型包括bfloat16（Atlas训练系列产品不支持）、float16、float32、float64、int32、int64。

    参数：
        - **input** (Tensor) - 输入Tensor。

    关键字参数：
        - **decimals** (int, 可选) - 要舍入到的小数位数（默认值：None）。如果为负数，则指定小数点左侧的位数。支持输入单元素Tensor转换为int。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
