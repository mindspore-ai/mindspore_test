mindspore.device_context.ascend.op_precision.conv_allow_hf32
============================================================

.. py:function:: mindspore.device_context.ascend.op_precision.conv_allow_hf32(value)

    是否为Conv类算子使能FP32转换为HF32。CANN默认使能Conv类算子FP32转换为HF32。
    如果您想了解更多详细信息，
    请查询 `昇腾社区 <https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/appdevgapi/aclcppdevg_03_1371.html/>`_ 了解。

    .. note::
        - 这是一个实验特性，可能会被更改或者删除。

    参数：
        - **value** (bool) - 是否为Conv类算子使能FP32转换为HF32。